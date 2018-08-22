from nltk.tag import StanfordNERTagger
from nltk.corpus import wordnet as wn
from question import Question
from collections import defaultdict
from multiprocessing import Process, Value, Lock

import util as ut
import nltk
import random
import json

# Mapping question words to most frequent NER tag from StanfordNER
# 'what', 'why', 'how' do not have a usual tag
ANSWER_TYPES = {'who':['PERSON'], 'what':[-1], 'when': ['DATE'],
'where':['LOCATION'],'why':[-1],'how':[-1], 'which':['ORGANIZATION']}

def extract_nps(sentence):
	'''
	input: `sentence` is a list of words corresponding to a sentence
	output: a list of lists of all NPs in `sentence`
	'''
	nps = []
	grammar = r"""
  		NP: {<DT|PP\$>?<JJ>*<NN>}   # taken from http://www.nltk.org/book/ch07.html
      		{<NNP>+}
	"""
	cp = nltk.RegexpParser(grammar)
	tagged_sent = nltk.pos_tag(sentence)
	parse_tree = cp.parse(tagged_sent)
	
	for subtree in parse_tree:
		if type(subtree) == nltk.tree.Tree:
			if subtree.label() == 'NP':
				removed_pos = []
				for leaf in subtree.leaves():
					removed_pos.append(leaf[0])
				nps.append(removed_pos)
	return nps

def collect_tags(tagged_sent, collection):
	'''
	input: 
		- `tagged_sent`: a list of (word,POS tag) pairs  
		- `collection`: {'LOCATION','ORGANIZATION','DATE','MONEY','PERSON','PERCENT','TIME'}
	output: a list of lists corresponding all contiguous words of type `collection`
	'''
	result = []
	found_tag = False
	
	for word, tag in tagged_sent:
		if found_tag and tag == collection:
			result[-1].append(word)
		if not found_tag and tag == collection:
			found_tag = True
			result.append([word])
		if tag != collection and found_tag:
			found_tag = False
	
	if result:
		return result
	else:
		return

def ner_tag(sentence):
	'''
	input: `sentence`: list of words corresponding to a sentence
	output: list of (word, NER tag) pairs 
	'''
	st = StanfordNERTagger('../english.muc.7class.distsim.crf.ser.gz',
		'../stanford-ner.jar')
	return st.tag(sentence)

def is_hyponym(check_word, word):
	'''
	input: 
		- `check_word`: a string
		- `word`: a string
	output: {True, False} depending on whether `check_word` exists as a recursive hyponym of `word`
	'''
	check_word = wn.synsets(check_word.lower())[0]
	word = wn.synsets(word.lower())[0]
	return check_word in set([i for i in word.closure(lambda s:s.hyponyms())])

def is_hypernym(check_word, word):
	'''
	input: 
		- `check_word`: a string
		- `word`: a string
	output: {True, False} depending on whether `check_word` exists as a recursive hypernym of `word`
	'''
	check_word = wn.synsets(check_word)[0]
	word = wn.synsets(word)[0]
	return check_word in set([i for i in word.closure(lambda s:s.hypernyms())])

def retrieve_top_docs(query, docs, similarity_measure, k=5):
	'''
	input:
		- `query`: a list of strings 
		- `docs`: a list of documents(sentences), alternativley a list of lists of strings
		- `similarity_measure`: function returning a score of similarity
		- `k`: an integer
	output: Top `k` documents(sentences) scored based on their similarity to `query`, 
			unless the highest-scored document scores above 0.3, in which case, only
			that document is returned
	'''
	all_words = ut.get_all_words(docs)
	all_terms = set(all_words)
	idf = ut.calculate_idf(docs,all_terms)
	sent_scores = [(doc, similarity_measure(query, doc, all_terms, idf)) for doc in docs]
	sent_scores = sorted(sent_scores, key=lambda t : t[1], reverse=True)
	sent_scores = [(sent, score) for sent, score in sent_scores if score > 0]
	
	if not sent_scores:
		return []
	elif sent_scores[0][1] >= 0.3: # top-ranked document >= 0.3, chose that one due to high correlation
		return sent_scores[0][0]
	else: # otherwise, simply keep the other sentences
		return [sentence for sentence,score in sent_scores[:k]]

def calculate_answer_window(document, focus, window_size=10):
	'''
	input: 
		- `document`: a list of strings corresponding to one document(sentence)
		- `focus`: a list of strings corresponding to the focus of a question
		- `window_size`: an integer
	output: returns the `left` and `right` indices of the window
	'''
	left = 0
	right = len(document)-1
	
	if not focus:
		return left,right
	try:
		attempted_left = document.index(focus[0][0]) - window_size
		if attempted_left >= left:
			left = attempted_left
		attempted_right = document.index(focus[-1][0]) + window_size
		if attempted_right <= right:
			right = attempted_right
	except:
		pass
	return left,right

def calculate_candidate_answers(question, docs):
	'''
	input:
		- `question`: a Question object
		- `docs`: a list of documents(sentences), alternatively a list of list of strings
	output: `all_answers`: returns a dictionary mapping document ids (position in list) to lists of 
		potential answers (list of strings)
	'''
	q_word = question.question_word
	focus  = question.focus
	all_answers = defaultdict(list)

	for i,doc in enumerate(docs):
		tagged_doc = ner_tag(doc)
		# check if q_word is not in answer_types
		if q_word in ANSWER_TYPES:
			for tag in ANSWER_TYPES[q_word]:
				if tag == -1:
					if q_word == 'how': # usually, how is followed by 'many' asking for a number
						all_answers[i] = [word for word in doc if word.isdigit()] # find digits
					else: # 'what' or 'why'
						# check if focus is hyper or hypnoym of words in doc
						for word in doc:
							if is_hypernym(focus[0],word) or is_hyponym(focus[0],word):
								all_answers[i].append(word)
							else:
								all_answers[i].append(random.choice(doc))
				else:
					answers = collect_tags(tagged_doc,tag)
					if answers:
						for answer in answers:
							all_answers[i].append(answer)
		else:
			# select random word in sentence
			return {i:random.choice(tagged_doc)[0]}
	return all_answers

def find_answer(question, answers, docs):
	'''
	input: 
		- `question`: a Question object
		- `answers`: a list of lists of strings
		- `docs`: a list of documents(sentences), alternatively a list of list of strings
	output: `final_answer`: a list of final answers
	'''
	focus  = question.focus
	final_answers = []
	for i in answers:
		left,right = calculate_answer_window(docs[i],focus)
		for answer in answers[i]:
			# if any answer component are in window, add to final results
			if any(map(lambda x: x in docs[i][left:right],answer)):
				final_answers.append(answer)
	return final_answers

def search(query, docs):
	'''
	input:
		- `query`: a string 
		- `docs`: a list of documents(sentences), alternatively a list of list of strings
	output: a string corresponding to the answer for the `query` in `docs`
	'''
	print query
	question = Question.classify_query(query)
	all_tagged_sents = [ner_tag(sent) for sent in docs]
	top_docs = retrieve_top_docs(question.context, docs, ut.cos_measure)
	candidates = calculate_candidate_answers(question, top_docs)
	if not candidates:
		if question.question_word in ANSWER_TYPES:
			answer_type = ANSWER_TYPES[question.question_word][0]
			potential_answers = [collect_tags(doc,answer_type) for doc in all_tagged_sents if doc is not None]
			answers_without_none = filter(lambda x: x is not None, potential_answers)
			if not answers_without_none:
				return random.choice(top_docs[0]) if top_docs else 'None'
			else:
				return ' '.join(random.choice(random.choice(answers_without_none)))
		else:
			return 'None'
	else:
		final_answers = find_answer(question, candidates, top_docs)
		return ' '.join(final_answers[0]) if final_answers else random.choice(top_docs[0])

def run_search(queries, docs, dictionary, count, total, lock):
	'''
	input:
		- `queries`: a list of lists of lists
			`queries`[i][j] is a list of words corresponding to the jth question in the ith context
		- `docs`: a list of lists of lists
			`docs`[i][j] returns a list of words corresponding to the jth sentence in the ith context
	output: `results`: a dictionary mapping question ids to the chosen answer
	'''
	assert len(queries) == len(docs)

	for i in range(len(queries)):
		for j,(query,id_) in enumerate(queries[i]):
			dictionary[id_] = search(query,docs[i])
			print str(count.value)+'/'+str(total) + " " + dictionary[id_]
			print '\n'
			with lock:
				count.value += 1

if __name__ == '__main__':
	docs = ut.collect_docs('../development.json')
	qs   = ut.collect_questions('../development.json')
	num_threads = 4
	amt = len(docs)/num_threads
	results = {}
	threads = []
	count = Value('i',1)
	lock = Lock()
	all_queries = []
	for context in qs:
		for q in context:
			all_queries.append(q)
	p1 = Process(target=run_search,args=(qs[:amt],docs[:amt],results,count,len(all_queries),lock))
	p2 = Process(target=run_search,args=(qs[amt:amt*2],docs[amt:amt*2],results,count,len(all_queries),lock))
	p3 = Process(target=run_search,args=(qs[amt*2:amt*3],docs[amt*2:amt*3],results,count,len(all_queries),lock))
	p4 = Process(target=run_search,args=(qs[amt*3:],docs[amt*3:],results,count,len(all_queries),lock))
	p1.start()
	p2.start()
	p3.start()
	p4.start()
	p1.join()
	p2.join()
	p3.join()
	p4.join()
	with open('../result.json','w') as f:
		json.dump(results,f)
