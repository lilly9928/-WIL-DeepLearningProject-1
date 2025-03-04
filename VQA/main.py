import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_loader import get_loader
from models import VqaModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():

    #input hyperparameter

    input_dir = 'D:/data/vqa/coco/simple_vqa'
    log_dir = './logs'
    model_dir='./models'
    max_qst_length = 30
    max_num_ans =10
    embed_size=64
    word_embed_size=300
    num_layers=2
    hidden_size=16
    learning_rate = 0.001
    step_size = 10
    gamma = 0.1
    num_epochs=30
    batch_size = 1024
    num_workers = 4
    save_step=1

    #log, model 디렉토리 만들기
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    #데이터 로더
    data_loader = get_loader(
        input_dir=input_dir,
        input_vqa_train='train.npy',
        input_vqa_valid='valid.npy',
        max_qst_length=max_qst_length,
        max_num_ans=max_num_ans,
        batch_size=batch_size,
        num_workers=num_workers)

    #데이터 로더에서 단어 수 , unk인덱스 가져옴
    qst_vocab_size = data_loader['train'].dataset.qst_vocab.vocab_size
    ans_vocab_size = data_loader['train'].dataset.ans_vocab.vocab_size
    ans_unk_idx = data_loader['train'].dataset.ans_vocab.unk2idx

    #vqa 모델
    model = VqaModel(
        embed_size=embed_size,
        qst_vocab_size=qst_vocab_size,
        ans_vocab_size=ans_vocab_size,
        word_embed_size=word_embed_size,
        num_layers=num_layers,
        hidden_size=hidden_size).to(device)

    #loss 함수
    criterion = nn.CrossEntropyLoss()

    params = list(model.img_encoder.model.model.fc.parameters()) \
        + list(model.qst_encoder.parameters()) \
        + list(model.fc1.parameters()) \
        + list(model.fc2.parameters())

    #옵티마이저, 스케줄러 정의
    optimizer = optim.Adam(params, lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    #train 시작
    for epoch in range(num_epochs):

        #PHASE ... ?
        for phase in ['train', 'valid']:
            #초기화
            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0
            batch_step_size = len(data_loader[phase].dataset) / batch_size

            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            #데이터로더에서 batch_idx, batch_sample가져옴
            for batch_idx, batch_sample in enumerate(data_loader[phase]):

                #batch_sample에서 image, question,label에 가져와서 값 정의
                image = batch_sample['image'].to(device).float()
                question = batch_sample['question'].to(device).long()
                label = batch_sample['answer_label'].to(device).long()
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                #zero_grad()함수 찾아보기
                optimizer.zero_grad()

                #torch.set_grad_enabled 찾아보기
                # train일 경우
                with torch.set_grad_enabled(phase == 'train'):
                    #모델에 이미지, 질문 넣어주고 output 추출
                    #pred 1, pred 2 추출
                    output = model(image, question)      # [batch_size, ans_vocab_size=1000]
                   # print('output',output)
                    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
                    _, pred_exp2 = torch.max(output, 1)  # [batch_size]

                   # print('pred_exp1', pred_exp1)
                  #  print('pred_exp2', pred_exp2)

                    loss = criterion(output, label)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #
                # for ans in multi_choice:
                #     print(ans)
                #     print(pred_exp1.cpu())
                #     print(ans == pred_exp1.cpu())

                # Evaluation metric of 'multiple choice'
                # Exp1: our model prediction to '<unk>' IS accepted as the answer.
                # Exp2: our model prediction to '<unk>' is NOT accepted as the answer.
                pred_exp2[pred_exp2 == ans_unk_idx] = -9999
                running_loss += loss.item()
                running_corr_exp1 += torch.stack([(ans == pred_exp1.cpu()) for ans in multi_choice]).any(dim=0).sum()
                running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()

                # Print the average loss in a mini-batch.
                if batch_idx % 100 == 0:
                    print('| {} SET | Epoch [{:02d}/{:02d}], Step [{:04d}/{:04d}], Loss: {:.4f}'
                          .format(phase.upper(), epoch+1, num_epochs, batch_idx, int(batch_step_size), loss.item()))

                    #step단계

            # Print the average loss and accuracy in an epoch.
            epoch_loss = running_loss / batch_step_size
            epoch_acc_exp1 = running_corr_exp1.double() / len(data_loader[phase].dataset)      # multiple choice
            epoch_acc_exp2 = running_corr_exp2.double() / len(data_loader[phase].dataset)      # multiple choice

            print('| {} SET | Epoch [{:02d}/{:02d}], Loss: {:.4f}, Acc(Exp1): {:.4f}, Acc(Exp2): {:.4f} \n'
                  .format(phase.upper(), epoch+1, num_epochs, epoch_loss, epoch_acc_exp1, epoch_acc_exp2))

            # Log the loss and accuracy in an epoch.
            with open(os.path.join(log_dir, '{}-log-epoch-{:02}.txt')
                      .format(phase, epoch+1), 'w') as f:
                f.write(str(epoch+1) + '\t'
                        + str(epoch_loss) + '\t'
                        + str(epoch_acc_exp1.item()) + '\t'
                        + str(epoch_acc_exp2.item()))

        # Save the model check points.
        if (epoch+1) % save_step == 0:
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict()},
                       os.path.join(model_dir, 'model-epoch-{:02d}.ckpt'.format(epoch+1)))

if __name__ == '__main__':
    main()
