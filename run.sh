
source activate transformers 
python main.py --train_file ../dataset/faq/train.csv --dev_file ../dataset/faq/dev.csv --output_dir model

python bert_main.py --data_dir ../dataset/faq/ --do_train --max_steps 128 --output_dir BERT_output  --train_batch_size 4

python faq_main.py --train_file ../dataset/faq/best_train.csv --dev_file ../dataset/faq/best_dev.csv --output_dir faq_model

python faq_predict.py --train_file ../dataset/faq/best_train.csv --output_dir faq_model

python faq_hinge.py --train_file ../dataset/faq/best_train.csv --dev_file ../dataset/faq/best_dev.csv --output_dir faq_hinge_model
