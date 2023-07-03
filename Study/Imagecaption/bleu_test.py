import nltk.translate.bleu_score as bleu

candidate = 'clinical information   mental change acute sah mainly noted along the basal cisterns associated ivh with mild hydrocephalus a large acute ich noted in right frontal base also small amount of sdh along bilateral convexities '
references = [
    'Clinical information  trauma Acute SAH along the basal cisterns and both cerebral sulci  right left small acute SDH along both frontal convexities and falx Small amount of pneumocephalus noted in left side cavernous sinus and T S sinuses  rather likely IV related air than trauma induced Recommend   Clinical correlation'
]

print('패키지 NLTK의 BLEU :',bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)),list(candidate.split()),weights=(0, 1, 0, 0)))