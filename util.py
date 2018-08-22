import nltk
import math
import json
import numpy as np

from collections import Counter

def text_to_vec(all_terms, words, idf):
	'''
	input: 
		- `all_terms`: a list of unique strings corresponding to all the words in the corpus
		- `words`: a list of strings for which the vector representation is calculated
		- `idf`: a dictionary mapping terms to their idf score
	output: a dictionary mapping the term to its tf-idf score
	'''
	if words:
		if isinstance(words[0],tuple):
			words = [word[0] for word in words] # remove POS tags when making vector
	d = {}
	for t in all_terms:
		d[t] = words.count(t) * idf[t]
	return d

def norm(d):
	'''
	input: `d`: a dictionary mapping terms to scores (a vector represenation)
	output: returns the normalized score of the vector representation
	'''
	sum_sq = 0
	for v in d:
		sum_sq += d[v] * d[v]
	return np.sqrt(sum_sq) if sum_sq > 0 else 0

def calculate_idf(docs, all_terms):
	'''
	input:
		- `docs`: a list of documents(sentences), alternatively a list of list of strings 
		- `all_terms`: a list of unique strings corresponding to all the words in the corpus
	output: `idf`: a dictionary mapping terms to their idf scores
	'''
	idf = {}
	N = len(docs)
	for t in all_terms:
		df = len([1 for sent in docs if t in sent])
		idf[t] = np.log(N / float(df + 1)) if N > 0 else 0
	return idf

def dot(d, q):
	'''
	input:
		- `d`: a vector (stored in dictionary form)
		- `q`: a vector (stored in dictionary form)
	output: the dot product of `d` and `q`
	'''
	sum=0
	for v in d:  # iterates through keys
		sum += d[v] * q[v]
	return sum

def cos_measure(query_words, sentence, all_terms, idf):
	'''
	input: 
		- `query_words`: a list of strings
		- `sentence`: a list of strings
		- `all_terms`: a list of unique strings corresponding to all the words in the corpus
		- `idf`: a dictionary mapping terms to their idf scores
	output: a score of the relationship between query_words and sentence, according to 
		cosine similarity
	'''
	try:
		A = text_to_vec(all_terms,query_words,idf)
		B = text_to_vec(all_terms,sentence,idf)
		result = dot(A, B) / float(norm(A) * norm(B))
	except:
		result = 0
	return result

def jaccard(query, document, all_terms, idf):
	'''
	input:
		- `query`: a list of strings
		- `document`: a list of strings
		- `all_terms`: unused
		- `idf`: unused
	output: Jaccard similarity score (# terms in common / # total number of terms)
	'''
	A = set(query)
	B = set(document)
	return len(A & B) / float(len(A | B))

def remove_stopwords(text):
	'''
	input: `text`: a list of strings
	output: a list of strings with all English stopwords removed
	'''
	stopwords = nltk.corpus.stopwords.words('english')
	return filter(lambda w: w.lower() not in stopwords, text)

def remove_pos_tags(tagged_text, stringify=False):
	'''
	input: `tagged_text`, list of (<word>,<tag>) pairs
	output: string of words joined by spaces
	'''
	words = []
	for word,tag in tagged_text:
		words.append(word)
	if stringify:
		return ' '.join(words)
	else:
		return words

def get_all_words(docs):
	'''
	input: `docs`: a list of documents(sentences), alternatively a list of list of strings
	output: `all_words`: a sorted list of all the words in `docs`
	'''
	all_words = []
	for doc in docs:
		for word in doc:
			all_words.append(word.lower())
	return sorted(all_words)

def collect_docs(file):
	'''
	input: `file`, a file name as a string
	output: `sentences`, a list of lists of lists
	sentences[i][j] returns a list of words corresponding to the 
		the jth sentence in the ith context
	'''
	sentences = []
	with open(file) as json_file:
		data = json.load(json_file)
	for entity in data['data']:
		for paragraph in entity['paragraphs']:
			words = []
			contexts = nltk.sent_tokenize(paragraph['context'])
			for context in contexts:
				words.append(nltk.word_tokenize(context.lower()))
			sentences.append(words)
	return sentences

def collect_questions(file):
	'''
	input: `file`, a file name as a string
	output: `questions`, a list of lists of lists
	questions[i][j] returns a list of words corresponding to the
		jth question in the ith context
	'''
	questions = []
	with open(file) as json_file:
		data = json.load(json_file)
	for entity in data['data']:
		for paragraph in entity['paragraphs']:
			question = []
			for q in paragraph['qas']:
				contexts = nltk.sent_tokenize(q['question'])
				for context in contexts:
					question.append((nltk.word_tokenize(context.lower()),q['id']))
			questions.append(question)
	return questions

def collect_answers(file):
	'''
	input: `file`, a file name as a string
	output: `answers`, a list of lists of lists
	answers[i][j] returns a list of answers corresponding to the
		jth answer group in the ith context
	'''
	answers = []
	with open(file) as json_file:
		data = json.load(json_file)
	for entity in data['data']:
		for paragraph in entity['paragraphs']:
			answer_group = []
			for qa in paragraph['qas']:
				answer = []
				for a in qa['answers']:
					answer.append(a['text'])
				answer_group.append(answer)
			answers.append(answer_group)
	return answers