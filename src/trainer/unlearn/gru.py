from typing import Any, Dict, Union
import torch
from torch import nn
from trainer.unlearn.base import UnlearnTrainer
import numpy as np

from trainer.utils import compute_dpo_loss
from trainer.unlearn.grad_diff import GradDiff


class GRU(GradDiff,UnlearnTrainer):
    def __init__(self,  gamma_gru=0.8, forget_loss_type="GradAscent", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gamma_gru = gamma_gru
        self.forget_loss_type = forget_loss_type
        self.gradient_accumulation_steps = kwargs["args"].gradient_accumulation_steps
        if self.ref_model is None and self.forget_loss_type == "NPO":
            self.ref_model = self._prepare_ref_model(self.model)

        # Initialization of internal variables to store gradients and computational states
        self.dotp_retain = 0.0
        self.flattened_gradient = 0.0
        self.flattened_retain = 0.0
        self.flattened_retain_prev = 0.0
        self.flattened_retain_accumulation = 0.0
        self.structure_map = None
        self.steps = 0
        self.gradient_accum = {}              
        self.retain_grad = {}

    def orthogonal_component(self, g, g1):
        """Compute the component of g orthogonal to g1."""
        g1g1 = self.compute_total_gradient_dot_product(g1, g1, self.structure_map)
        gg1 = self.dotp_retain
        projection = gg1/g1g1* g1
        orthogonal = g - projection

        return orthogonal
    
    def store_grads(self, model, loss=None, typ=None):
        """
        Captures and stores gradients instead of applying them directly within the training loop. This method
        allows for sophisticated gradient manipulations before they are used to update the model, substituting
        the portion of `training_step` where gradients would typically be computed and immediately applied.

        Args:
            model (torch.nn.Module): The model from which to store gradients.
            loss (torch.Tensor, optional): The loss tensor to perform backward operation. If provided, will compute gradients.
        """

        # Perform backward pass if a loss tensor is provided
        if loss:

            # if self.args.n_gpu > 1:
            #     loss = loss.mean() 

            loss = loss / self.gradient_accumulation_steps
            loss.backward() # Compute gradients

        # Loop through parameters and accumulate gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)

                # Choose the correct dictionary based on 'typ'
                if typ == "objective":
                    target_dict = self.gradient_accum
                elif typ == "retain":
                    target_dict = self.retain_grad
                else:
                    raise ValueError("Invalid type specified for gradient storage")

                # Initialize the dictionary key if it doesn't exist
                if name not in target_dict:
                    target_dict[name] = torch.zeros_like(param.grad, device=param.grad.device)  # Initialize on the same device

                # Accumulate the gradients
                target_dict[name] += param.grad.detach()

        if loss:
            model.zero_grad() # Clear gradients after storage

    def flatten2cpu(self):
        """
        Flattens accumulated gradients from different gradient dictionaries, moves them to CPU,
        and stores them along with a structure map for each type of gradient.
        """

        # Helper function to flatten gradients, move to CPU, and record their original structure
        def flatten_to_cpu_and_record_structure(gradient_dict):
            flattened_grads = []
            structure_map = []
            for name, grad in gradient_dict.items():
                if grad is not None:
                    grad_flat = grad.view(-1)
                    flattened_grads.append(grad_flat)
                    structure_map.append((name, grad.shape))

            if flattened_grads:
                return torch.cat(flattened_grads).to('cpu'), structure_map
            else:
                return torch.tensor([], dtype=torch.float32).to('cpu'), []

  
        self.flattened_gradient, self.structure_map = flatten_to_cpu_and_record_structure(self.gradient_accum)      
        self.flattened_retain_accumulation, _ = flatten_to_cpu_and_record_structure(self.retain_grad)

    def compute_total_gradient_dot_product(self, flattened_grads1, flattened_grads2, structure_map):
        """
        Computes the total dot product between gradients from two sets of flattened gradients and their respective structure maps.

        Args:
            flattened_grads1 (torch.Tensor): The first flattened gradient tensor.
            flattened_grads2 (torch.Tensor): The second flattened gradient tensor.
            structure_map (list): A list of tuples containing parameter names and their corresponding shapes for the second set of gradients.

        Returns:
            float: The total dot product summed across all matching layers.
        """
        #assert len(structure_map1) == len(structure_map2), "Both gradient structures must contain the same number of elements."

        total_dot_product = 0.0
        index = 0

        # Ensure both gradient tensors are on the same device
        flattened_grads1 = flattened_grads1.to('cuda')
        flattened_grads2 = flattened_grads2.to('cuda')

        for ((name1, shape1), (name2, shape2)) in zip(structure_map, structure_map):
            assert shape1 == shape2, f"Gradient mismatch: {name1} vs {name2} or {shape1} vs {shape2}"

            size = np.prod(shape1)  # Total number of elements in this layer's gradient
            grad_slice1 = flattened_grads1[index:index + size].view(shape1)
            grad_slice2 = flattened_grads2[index:index + size].view(shape2)

            # Compute the dot product of the two gradient slices
            dot_product = (grad_slice1 * grad_slice2).sum()
            total_dot_product += dot_product.item()

            index += size

        return total_dot_product

    def restore_gradients(self, model):
        """
        Restores gradients to the model's parameters directly from self.flattened_gradient.

        """
        index = 0  # Index to track position in the flattened gradient tensor

        for name, shape in self.structure_map:
            size = np.prod(shape)  # Total number of elements in this gradient
            if size == 0:  # Skip layers with no parameters
                continue

            # Extract the relevant slice from the flattened gradient tensor
            grad_slice = self.flattened_gradient[index:index + size].view(shape)

            # Find the corresponding parameter in the model
            param = next((p for n, p in model.named_parameters() if n == name), None)
            if param.requires_grad:
                # Check if the shape of the extracted gradient matches the parameter's shape
                if grad_slice.shape != param.shape:
                    raise ValueError(f"Gradient shape mismatch for {name}: expected {param.shape}, got {grad_slice.shape}")

                # Set the parameter's gradient to the extracted slice
                param.grad = grad_slice.to(param.device)

            index += size  # Update index to the start of the next gradient slice

        if index != self.flattened_gradient.numel():
            raise ValueError("Total number of gradient elements does not match the length of the flattened gradient tensor.")
       
    def pipeline(self):
        if self.dotp_retain < 0:
            self.flattened_gradient = self.orthogonal_component(self.flattened_gradient, self.flattened_retain)
            torch.cuda.empty_cache()

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        retain_loss += retain_outputs.loss
        return retain_loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """Overridden training_step to include custom GRU logic.

                Notes:
                    - Gradient computation via backward pass has already been performed by `store_grads`. 
                    - This method performs additional operations on the stored gradients, including flattening gradients, smoothing retain gradients via EMA, and adjusting 
                    gradients by projection.
                    - After these custom manipulations, modified gradients are restored back to model parameters before optimization.
            
            """
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            del inputs
            torch.cuda.empty_cache()

            if self.steps % self.gradient_accumulation_steps == 0:

                # Flatten and move accumulated gradients to CPU before clearing
                self.flatten2cpu()
                self.gradient_accum = {}
                self.retain_grad = {}

                # For Stable Estimation
                self.flattened_retain = self.gamma_gru * self.flattened_retain_accumulation + (1 - self.gamma_gru) * self.flattened_retain_prev
                self.flattened_retain_prev = self.flattened_retain
 
                self.dotp_retain = self.compute_total_gradient_dot_product(self.flattened_gradient, self.flattened_retain, self.structure_map)
                self.pipeline()

                self.restore_gradients(model)
                self.flattened_retain_accumulation = 0
                torch.cuda.empty_cache()

            return loss.detach() / self.args.gradient_accumulation_steps
   
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.forget_loss_type == "GradAscent":
        
            forget_inputs = inputs["forget"]
            forget_inputs = {
                "input_ids": forget_inputs["input_ids"],
                "attention_mask": forget_inputs["attention_mask"],
                "labels": forget_inputs["labels"],
            }

            forget_outputs = model(**forget_inputs)
            forget_loss = -forget_outputs.loss
            del forget_outputs
            self.store_grads(model, loss=forget_loss, typ = "objective")

            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            self.store_grads(model, loss=retain_loss, typ = "retain")

            loss = forget_loss
            self.steps +=1

        elif self.forget_loss_type == "NPO":

            forget_inputs = inputs["forget"]
            forget_loss, forget_outputs = compute_dpo_loss(
                model=model,
                ref_model=self.ref_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=0.1,
            )
            del forget_outputs
            self.store_grads(model, loss=forget_loss, typ = "objective")

            retain_inputs = inputs["retain"]
            retain_inputs = {
                "input_ids": retain_inputs["input_ids"],
                "attention_mask": retain_inputs["attention_mask"],
                "labels": retain_inputs["labels"],
            }
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
            self.store_grads(model, loss=retain_loss, typ = "retain")

            loss = forget_loss + retain_loss


        return (loss, forget_outputs) if return_outputs else loss