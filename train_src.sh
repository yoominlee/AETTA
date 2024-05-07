  LOG_PREFIX="reproduce_src"

  DATASETS="es" #"imagenet" # cifar10 or cifar100
  METHODS="Src"

  echo DATASETS: $DATASETS
  echo METHODS: $METHODS

  GPUS=(1) #available gpus
  NUM_GPUS=${#GPUS[@]}

  sleep 1 # prevent mistake
  mkdir raw_logs # save console outputs here

  #### Useful functions
  wait_n() {
    #limit the max number of jobs as NUM_MAX_JOB and wait
    background=($(jobs -p))
    local default_num_jobs=8 #num concurrent jobs
    local num_max_jobs=${1:-$default_num_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
      wait -n
    fi
  }

  ###############################################################
  ##### Source Training; Source Evaluation: Source domains  #####
  ###############################################################
  train_source_model() {
    echo train_source_model'('')' START
    i=0
    update_every_x="64"
    memory_size="64"
    for DATASET in $DATASETS; do
      for METHOD in $METHODS; do

        validation="--dummy"

        if [ "${DATASET}" = "cifar10" ] || [ "${DATASET}" = "cifar10outdist" ]; then
          EPOCH=200
          MODEL="resnet18"
          TGT="test"
        elif [ "${DATASET}" = "cifar100" ]; then
          EPOCH=200
          MODEL="resnet18"
          TGT="test"
        elif [ "${DATASET}" = "imagenet" ]; then
          EPOCH=30
          MODEL="resnet18_pretrained"
          TGT="test"
        elif  [ "${DATASET}" = "es" ]; then ###################             <---
          EPOCH=20 # git/ImageNet-ES/configs/tin_config.py 참고
          MODEL="resnet18"
          TGT="test"
        fi

        for SEED in 0 1 2; do
          if [[ "$METHOD" == *"Src"* ]]; then
            #### Train with BN
            for tgt in $TGT; do
              echo right before main.py @train_src.sh
              # echo GPU: $GPUS
              # echo DATASET: $DATASET
              # echo tgt: $tgt
              # echo MODEL: $MODEL
              # echo EPOCH: $EPOCH
              # echo update_every_x: $update_every_x
              # echo memory_size: $memory_size 
              # echo SEED: $SEED
              # echo validation: $validation
              # echo LOG_PREFIX: $LOG_PREFIX


              python main.py --gpu_idx ${GPUS[i % ${NUM_GPUS}]} --dataset ${DATASET} --method Src --tgt ${tgt} --model $MODEL --epoch $EPOCH --update_every_x ${update_every_x} --memory_size ${memory_size} --seed $SEED \
                --log_prefix ${LOG_PREFIX}_${SEED} \
                ${validation} \
                --num_workers 1 \
                2>&1 | tee raw_logs/${DATASET}_${LOG_PREFIX}_${SEED}_job${i}.txt &

              i=$((i + 1))
              wait_n
            done
          fi
        done
      done
    done

    wait
  }

  train_source_model
