import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from help import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='D:/data/vqa/coco/simple_vqa/Annotations/annotations/caption/dataset_coco.json',
                       image_folder='D:/data/vqa/coco/simple_vqa/Images',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='D:/data/vqa/coco/simple_vqa/cococaption',
                       max_len=50)