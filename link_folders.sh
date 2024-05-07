#!/bin/bash

# 원본 디렉토리 경로
SOURCE_DIR="/home/datasets/ILSVRC12/train"
# 목표 디렉토리 경로
TARGET_DIR="$HOME/git/AETTA/dataset/ImageNet-C/origin/Data/CLS-LOC/train"

# 목표 디렉토리가 없다면 생성
mkdir -p "$TARGET_DIR"

# 원본 디렉토리 내의 모든 폴더에 대해 반복
for folder in "$SOURCE_DIR"/*; do
  # 폴더만 선택
  if [ -d "$folder" ]; then
    # 링크 생성을 위한 폴더명 추출
    folder_name=$(basename "$folder")
    # 심볼릭 링크 생성
    ln -s "$folder" "$TARGET_DIR/$folder_name"
  fi
done

echo "All folders have been linked successfully."
