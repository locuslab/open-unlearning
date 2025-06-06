#!/bin/bash



########################################################################################################################
########################################### Final best parameters #####################################################
########################################################################################################################
# for an 8 GPU system:
num_processes=8


##############################################  TOFU #####################################################
per_device_train_batch_size=4
learning_rate=0.00001
dual_warmup_epochs=5

pref=100
dual_step_size=5
retain_loss_eps=0.3

retain_precentages=(90 95 99)
models=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct gemma-7b-it)

for model in "${models[@]}"; do
  for retain_percentage in "${retain_precentages[@]}"; do

    if [ "$retain_percentage" = "90" ]; then
      forget_split=forget10
      retain_split=retain90
    elif [ "$retain_percentage" = "95" ]; then
      forget_split=forget05
      retain_split=retain95
    elif [ "$retain_percentage" = "99" ]; then
      forget_split=forget01
      retain_split=retain99
    else
    #  echo "hello"
      echo "Invalid retain percentage. Please set it to 90, 95, or 99."
      exit 1
    fi


    if [ "$model" = "Llama-3.2-1B-Instruct" ]; then
      pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
      num_train_epochs=10
    elif [ "$model" = "Llama-3.2-3B-Instruct" ]; then
      pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.2-3B-Instruct_full
      num_train_epochs=10
    elif [ "$model" = "Llama-3.1-8B-Instruct" ]; then
      pretrained_model_name_or_path=open-unlearning/tofu_Llama-3.1-8B-Instruct_full
      num_train_epochs=30
    elif [ "$model" = "gemma-7b-it" ]; then
      pretrained_model_name_or_path=tamarsonha/TOFU-target-gemma-7b-it
      num_train_epochs=20
    else
      echo "Invalid model name. Please set it to Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, or gemma-7b-it."
      exit 1
    fi

    task_name=PDU-TOFU$retain_split-E$num_train_epochs-lr$learning_rate-P1-$pref-Primal$retain_loss_eps-Step$dual_step_size-Warmup$dual_warmup_epochs-model_$model
    accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=$num_processes \
        src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default \
        forget_split=$forget_split retain_split=$retain_split\
        trainer=PDU\
        trainer.args.num_train_epochs=$num_train_epochs\
        trainer.args.eval_on_start=false trainer.args.do_eval=false\
        trainer.args.per_device_train_batch_size=$per_device_train_batch_size\
        trainer.args.learning_rate=$learning_rate\
        trainer.method_args.preferences=[1.,$pref]\
        trainer.method_args.primal_dual=true trainer.method_args.retain_loss_eps=$retain_loss_eps\
        trainer.method_args.dual_step_size=$dual_step_size\
        trainer.method_args.dual_update_upon="step" trainer.method_args.dual_warmup_epochs=$dual_warmup_epochs\
        task_name=$task_name\
        model=$model model.model_args.pretrained_model_name_or_path=$pretrained_model_name_or_path
  done
done


######################################################## MUSE #########################################################
dual_step_size=1
num_train_epochs=10
dual_warmup_epochs=3
data_splits=("News" "Books")
learning_rate=0.00001
dual_update_upon="step"

models=(Llama-2-7b-hf Llama-2-13b-hf)
pref=50

for model in "${models[@]}"; do
  for data_split in "${data_splits[@]}"; do

    if [ "$model" = "Llama-2-7b-hf" ]; then
      pretrained_model_name_or_path=muse-bench/MUSE-${data_split}_target
      epsNews=(1.5)
      epsBooks=(0.1)
    elif [ "$model" = "Llama-2-13b-hf" ]; then
      pretrained_model_name_or_path=tamarsonha/MUSE-${data_split}-target-Llama-2-13b-hf
      epsNews=(0.8)
      epsBooks=(0.6)
    else
      exit 1
    fi


    if [ "$data_split" == "News" ]; then
      eps_array=("${epsNews[@]}")
    else
      eps_array=("${epsBooks[@]}")
    fi

    for retain_loss_eps in "${eps_array[@]}"; do
      task_name=PDU-Muse$data_split-E$num_train_epochs-lr$learning_rate-P1-$pref-Primal$retain_loss_eps-Step$dual_step_size-Warmup$dual_warmup_epochs-model$model
      accelerate launch --config_file configs/accelerate/default_config.yaml --num_processes=$num_processes \
          src/train.py --config-name=unlearn.yaml experiment=unlearn/muse/default \
          data_split=$data_split\
          trainer=PDU\
          trainer.args.num_train_epochs=$num_train_epochs\
          trainer.args.eval_on_start=false trainer.args.do_eval=false\
          trainer.args.per_device_train_batch_size=$per_device_train_batch_size\
          trainer.args.learning_rate=$learning_rate\
          trainer.method_args.preferences=[1.,$pref]\
          trainer.method_args.primal_dual=true trainer.method_args.retain_loss_eps=$retain_loss_eps\
          trainer.method_args.dual_step_size=$dual_step_size\
          trainer.method_args.dual_update_upon="step" trainer.method_args.dual_warmup_epochs=$dual_warmup_epochs\
          task_name=$task_name\
          model=$model model.model_args.pretrained_model_name_or_path=$pretrained_model_name_or_path
    done
  done
done
