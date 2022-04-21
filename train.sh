#python generate.py ../document_level/de-en/use_euclidean_destdir/ --path ../document_level/de-en/use_euclidean_model/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --bert-model-name bert-base-german-cased > ../document_level/de-en/use_euclidean_model/bleu_score.txt

#python generate.py ../document_level/de-en/use_manhattan_destdir/ --path ../document_level/de-en/use_manhattan_model/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --bert-model-name bert-base-german-cased > ../document_level/de-en/use_manhattan_model/bleu_score.txt

#python generate.py ../document_level/es-en/use_cosine_destdir/ --path ../document_level/es-en/use_cosine_model/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --bert-model-name bert-base-multilingual-cased > ../document_level/es-en/use_cosine_model/bleu_score.txt

#python generate.py ../document_level/es-en/use_euclidean_destdir/ --path ../document_level/es-en/use_euclidean_model/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --bert-model-name bert-base-multilingual-cased > ../document_level/es-en/use_euclidean_model/bleu_score.txt

python train.py ../document_level/es-en/use_manhattan_destdir/ -a transformer_s2_iwslt_de_en --optimizer adam --adam-betas '(0.9,0.98)' --lr 0.0005 -s es -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --warmup-updates 4000 --warmup-init-lr '1e-07' --save-dir ../document_level/es-en/use_manhattan_model/ --share-all-embeddings --encoder-bert-dropout --encoder-bert-dropout-ratio 0.5 --bert-model-name bert-base-multilingual-cased --max-epoch 60 | tee -a ../document_level/es-en/use_manhattan_model/log.txt
#python generate.py ../document_level/de-en/use_manhattan_destdir/ --path ../document_level/de-en/use_manhattan_model/checkpoint_best.pt --beam 5 --batch-size 128 --remove-bpe --bert-model-name bert-base-german-cased > ../document_level/de-en/use_manhattan_model/bleu_score.txt


