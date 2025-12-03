"""
SIBL (Sparse Bilevel Augmented Lagrangian) Unlearning Method
==============================================================

Implements the S-BiAL algorithm for machine unlearning with sparsity constraints.
Uses bilevel optimization with Augmented Lagrangian method and implicit differentiation.
"""

import torch
import torch.nn as nn
import time
import copy
import logging
from typing import Dict, Optional
from trainer.unlearn.base import UnlearnTrainer
from trainer.sparsity import SparsityManager

logger = logging.getLogger(__name__)

try:
    from scipy.sparse.linalg import LinearOperator, cg
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available, will use PyTorch CG implementation")


class SIBL(UnlearnTrainer):
    """Sparse Bilevel Augmented Lagrangian for TOFU Unlearning."""

    def __init__(
        self,
        # S-BiAL specific parameters
        use_sparsity: bool = True,
        sparsity: float = 0.9,
        sparsity_method: str = "layerwise_magnitude",
        epsilon: float = 0.1,  # Retain loss budget
        T: int = 20,  # Number of outer iterations
        K: int = 10,  # Number of inner iterations
        eta_theta: float = 1e-4,  # Outer learning rate
        eta_in: float = 1e-4,  # Inner learning rate
        rho: float = 1.0,  # Penalty parameter for AL
        gamma: float = 1e-4,  # L1 regularization coefficient
        use_implicit: bool = True,  # Use implicit differentiation
        cg_iters: int = 10,  # Conjugate gradient iterations
        cg_tol: float = 1e-3,  # CG tolerance
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Store configuration
        self.use_sparsity = use_sparsity
        self.sparsity = sparsity
        self.sparsity_method = sparsity_method
        self.epsilon = epsilon
        self.T = T
        self.K = K
        self.eta_theta = eta_theta
        self.eta_in = eta_in
        self.rho = rho
        self.gamma = gamma
        self.use_implicit = use_implicit
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol

        # Initialize dual variable
        self.lambda_dual = 0.0

        # History tracking
        self.history = {
            'iter': [],
            'L_forget': [],
            'L_retain': [],
            'residual': [],
            'lambda': [],
            'time': []
        }

        # Initialize sparsity mask (will be created when training starts)
        self.mask_dict = None

    def _initialize_mask(self):
        """Initialize sparsity mask for the model."""
        if self.use_sparsity:
            logger.info(f"Creating sparsity mask with {self.sparsity_method} "
                       f"at {self.sparsity} sparsity...")
            self.mask_dict = SparsityManager.create_mask(
                self.model,
                sparsity=self.sparsity,
                method=self.sparsity_method,
                device=self.args.device
            )
        else:
            # No sparsity: all ones mask
            logger.info("No sparsity constraints - using full model")
            self.mask_dict = {
                name: torch.ones_like(param.data).to(self.args.device)
                for name, param in self.model.named_parameters()
            }

    def compute_forget_loss(self, batch):
        """Compute forget loss using logit margin flattening."""
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Logit margin flattening
        logits = outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        margins = max_logits - mean_logits
        return margins.mean()

    def compute_retain_loss(self, batch):
        """Compute retain loss (standard cross-entropy)."""
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch.get('labels', input_ids).to(self.args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss

    def compute_sparsity_regularizer(self):
        """Compute L1 sparsity regularizer."""
        reg = 0.0
        for name, param in self.model.named_parameters():
            if name in self.mask_dict:
                mask = self.mask_dict[name]
                reg += (param.abs() * mask).sum()
        return self.gamma * reg

    def inner_step(self, batch):
        """Single inner optimization step on retain set."""
        self.model.train()

        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch.get('labels', input_ids).to(self.args.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Add L1 sparsity regularizer
        l1_reg = sum(
            (p.abs() * self.mask_dict[name]).sum()
            for name, p in self.model.named_parameters()
            if name in self.mask_dict and p.requires_grad
        )
        loss_total = loss + self.gamma * l1_reg

        loss_total.backward()

        # Masked gradient update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.mask_dict:
                    param.data.sub_(self.eta_in * param.grad * self.mask_dict[name])
                param.grad = None

        return loss.item()

    def inner_loop(self, retain_loader):
        """Inner loop: Optimize on retain set."""
        retain_iter = iter(retain_loader)

        for k in range(self.K):
            try:
                batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                batch = next(retain_iter)

            self.inner_step(batch)

    def flatten_params(self, params_list):
        """Flatten list of parameters to single vector."""
        return torch.cat([p.reshape(-1) for p in params_list])

    def unflatten_params(self, flat_vec, params_list):
        """Unflatten vector back to parameter shapes."""
        unflattened = []
        offset = 0
        for param in params_list:
            numel = param.numel()
            unflattened.append(flat_vec[offset:offset+numel].reshape(param.shape))
            offset += numel
        return unflattened

    def flatten_mask(self):
        """Flatten all masks to single vector."""
        masks = []
        for name, param in self.model.named_parameters():
            if name in self.mask_dict:
                masks.append(self.mask_dict[name].reshape(-1))
            else:
                masks.append(torch.ones(param.numel(), device=self.args.device))
        return torch.cat(masks)

    def compute_hvp(self, loss, params, v):
        """Compute Hessian-vector product H*v."""
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        grad_v = (flat_grad * v).sum()

        hvp_grads = torch.autograd.grad(grad_v, params, retain_graph=True)
        flat_hvp = torch.cat([h.reshape(-1) for h in hvp_grads])

        return flat_hvp

    def conjugate_gradient_scipy(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient (scipy implementation)."""
        # Store original device and dtype
        device = b.device
        dtype = b.dtype
        n = b.numel()

        # Convert b to numpy
        b_np = b.detach().cpu().numpy()

        # Define the matrix-vector product function for LinearOperator
        def matvec(v):
            # Convert numpy array to torch tensor
            v_torch = torch.from_numpy(v).to(device=device, dtype=dtype)
            # Compute Hessian-vector product
            Hv_torch = hvp_func(v_torch)
            # Convert back to numpy
            return Hv_torch.detach().cpu().numpy()

        # Create LinearOperator
        A = LinearOperator((n, n), matvec=matvec, dtype=b_np.dtype)

        # Solve using scipy's CG
        x_np, info = cg(A, b_np, maxiter=self.cg_iters, atol=self.cg_tol, rtol=0)

        # Convert solution back to torch tensor
        x = torch.from_numpy(x_np).to(device=device, dtype=dtype)

        # Log convergence info
        if info == 0:
            logger.debug("CG converged successfully")
        elif info > 0:
            logger.debug(f"CG did not converge, {info} iterations reached")
        else:
            logger.warning("CG illegal input or breakdown")

        return x

    def conjugate_gradient_torch(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient (PyTorch implementation)."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.sum(r * r)

        for i in range(self.cg_iters):
            Ap = hvp_func(p)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)

            if torch.sqrt(rs_new) < self.cg_tol:
                logger.debug(f"CG converged at iteration {i}")
                break

            beta = rs_new / (rs_old + 1e-10)
            p = r + beta * p
            rs_old = rs_new

        return x

    def conjugate_gradient(self, hvp_func, b):
        """Solve H*x = b using Conjugate Gradient."""
        if SCIPY_AVAILABLE:
            return self.conjugate_gradient_scipy(hvp_func, b)
        else:
            return self.conjugate_gradient_torch(hvp_func, b)

    def outer_step(self, forget_batch, retain_batch):
        """Outer loop: Update parameters to forget while respecting budget."""
        self.model.train()

        # Compute forget loss
        forget_ids = forget_batch['input_ids'].to(self.args.device)
        forget_mask = forget_batch['attention_mask'].to(self.args.device)

        forget_outputs = self.model(input_ids=forget_ids, attention_mask=forget_mask)
        logits = forget_outputs.logits
        max_logits = logits.max(dim=-1)[0]
        mean_logits = logits.mean(dim=-1)
        L_fgt = (max_logits - mean_logits).mean()

        # Compute retain loss
        retain_ids = retain_batch['input_ids'].to(self.args.device)
        retain_mask = retain_batch['attention_mask'].to(self.args.device)
        retain_labels = retain_batch.get('labels', retain_ids).to(self.args.device)

        retain_outputs = self.model(
            input_ids=retain_ids,
            attention_mask=retain_mask,
            labels=retain_labels
        )
        L_ret = retain_outputs.loss

        # Constraint residual
        r = L_ret.item() - self.epsilon

        # Augmented Lagrangian objective
        L_alm = L_fgt + self.lambda_dual * L_ret + 0.5 * self.rho * r**2

        # Compute raw gradient
        L_alm.backward(retain_graph=self.use_implicit)

        # Store raw gradients
        g_alm_dict = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                g_alm_dict[name] = param.grad.clone()
            else:
                g_alm_dict[name] = torch.zeros_like(param.data)

        # Clear gradients
        self.model.zero_grad()

        # Implicit correction (if enabled)
        if self.use_implicit:
            L_ret_v = self.compute_retain_loss(retain_batch)
            R_theta_v = self.compute_sparsity_regularizer()
            L_inner = L_ret_v + R_theta_v

            params_list = [p for p in self.model.parameters() if p.requires_grad]

            grads_inner = torch.autograd.grad(
                L_inner, params_list,
                create_graph=True, retain_graph=True
            )

            mask_flat = self.flatten_mask()
            v = torch.cat([g.reshape(-1) for g in grads_inner]) * mask_flat

            def hvp_func(vec):
                vec_masked = vec * mask_flat
                hvp = self.compute_hvp(L_inner, params_list, vec_masked)
                return hvp * mask_flat

            h = self.conjugate_gradient(hvp_func, v)

            hvp_correction = self.compute_hvp(L_alm, params_list, h)

            correction_unflattened = self.unflatten_params(hvp_correction, params_list)

            for (name, param), correction in zip(
                self.model.named_parameters(), correction_unflattened
            ):
                if name in g_alm_dict:
                    g_alm_dict[name] = g_alm_dict[name] - correction

        # Primal update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.mask_dict and name in g_alm_dict:
                    mask = self.mask_dict[name]
                    param.data.sub_(self.eta_theta * g_alm_dict[name] * mask)
                param.grad = None

        # Dual update
        self.lambda_dual = max(0.0, self.lambda_dual + self.rho * r)

        return L_fgt.item(), L_ret.item(), r

    def train(self):
        """Override the train method to implement custom S-BiAL training loop."""
        # Initialize mask
        if self.mask_dict is None:
            self._initialize_mask()

        # Get data loaders
        train_dataloader = self.get_train_dataloader()

        logger.info(f"\nStarting S-BiAL unlearning for {self.T} iterations...")
        logger.info(f"Retain budget ε = {self.epsilon:.4f}")
        logger.info(f"Use implicit correction: {self.use_implicit}")

        for t in range(self.T):
            t_start = time.time()

            # Get data iterators
            data_iter = iter(train_dataloader)

            # Get forget and retain batches
            try:
                combined_batch = next(data_iter)
                forget_batch = combined_batch['forget']
                retain_batch = combined_batch['retain']
            except (StopIteration, KeyError) as e:
                logger.error(f"Error getting batches: {e}")
                break

            # Inner loop
            # Create a simple retain loader from the current batch
            retain_loader = [retain_batch] * self.K

            self.inner_loop(retain_loader)

            # Outer loop
            L_fgt, L_ret, r = self.outer_step(forget_batch, retain_batch)

            # Track history
            t_elapsed = time.time() - t_start
            self.history['iter'].append(t)
            self.history['L_forget'].append(L_fgt)
            self.history['L_retain'].append(L_ret)
            self.history['residual'].append(r)
            self.history['lambda'].append(self.lambda_dual)
            self.history['time'].append(t_elapsed)

            # Log progress
            if t % 2 == 0 or t == self.T - 1:
                logger.info(
                    f"[{t:3d}/{self.T}] L_fgt={L_fgt:.4f} | L_ret={L_ret:.4f} | "
                    f"r={r:+.4f} | λ={self.lambda_dual:.3f} | t={t_elapsed:.2f}s"
                )

            # Update training state
            self.state.global_step = t + 1

            # Evaluation and checkpointing
            if self.args.evaluation_strategy != "no" and (t + 1) % self.args.eval_steps == 0:
                self.evaluate()

            if self.args.save_strategy != "no" and (t + 1) % self.args.save_steps == 0:
                self.save_model()

        logger.info("Unlearning complete!")

        # Return training output
        return type('TrainOutput', (), {'global_step': self.T, 'training_loss': L_ret})()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss for evaluation.
        This is only used during evaluation, not during training.
        """
        # For evaluation, just compute standard loss
        if 'forget' in inputs:
            forget_inputs = inputs['forget']
            forget_inputs = {
                'input_ids': forget_inputs['input_ids'],
                'attention_mask': forget_inputs['attention_mask'],
                'labels': forget_inputs.get('labels', forget_inputs['input_ids']),
            }
            outputs = model(**forget_inputs)
            loss = outputs.loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
