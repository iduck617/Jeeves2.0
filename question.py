import util as ut
import nltk 

class Question(object):
	def __init__(self, question_word, focus, context):
		self.question_word = question_word
		self.focus = focus
		self.context = context
	
	def __str__(self):
		return self.question_word + '\n' + str(self.focus) +  '\n' + str(self.context)
	
	@staticmethod
	def classify_query(query_words):
		'''
		input: `query_words`: a list of strings corresponding to a question
		ouput: a Question object
		'''
		question_word = query_words[0]
		query_no_stops = ut.remove_stopwords(query_words)
		found_np = False
		focus = []
		nouns = set(['NN','NNS','NNP','NNPS'])
		for word,tag in nltk.pos_tag(query_words)[1:]:
			if tag in nouns:
				found_np = True
			if found_np and tag in nouns:
				focus.append((word,tag))
			if tag not in nouns and found_np:
				break
		return Question(question_word.lower(), focus, query_no_stops)




