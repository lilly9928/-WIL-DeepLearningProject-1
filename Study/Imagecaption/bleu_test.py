import nltk.translate.bleu_score as bleu

candidate = 'a woman with a pink umbrella is walking down a street'
references = [
    'Woman in swim suit holding parasol on sunny day',
    'A woman posing for the camera holding a pink open umbrella and wearing a bright floral ruched bathing suit by a life guard stand with lake green trees and a blue sky with a few clouds behind',
    'A woman in a floral swimsuit holds a pink umbrella',
     'A woman with an umbrella near the sea',
     'A girl in a bathing suit with a pink umbrella'
]

print('패키지 NLTK의 BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),list(candidate.split()),weights=(1, 0, 0, 0)))