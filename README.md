# FAQ问答模型

- main.py 针对一个问题，和一个可能的答案，判断该答案是不是最佳答案。一个二分类问题。其中实现了两种loss的训练，cross entropy loss和cosine embedding loss。
- models.py 一个简单的基于GRU的dual encoder model。
- faq_hinge.py 使用hinge loss训练faq model。这里计算了问题和回复之间的某种相似度。
- faq_main.py 使用cross entropy或者cosine embedding loss做模型训练。
- faq_predict.py 使用faq模型预测答案。
