import json
import sys
import util

def chose_word(query,answer):
	mid = len(answer)/2
	return answer[mid] if answer else ''

def run_baseline(queries, docs):
	assert len(queries) == len(docs)
	results = {} # map ids to answers
	for i in range(len(queries)):
		for query,id_ in queries[i]:
			response = util.search(query,docs[i])
			answer = response[0][0] if response else []
			results[id_] = chose_word(query,answer)
	return results

def run_search(queries, docs):
	assert len(queries) == len(docs)
	results = {} # map ids to answers
	for i in range(len(queries)):
		for query,id_ in queries[i]:
			response = util.search(query,docs[i])
			answer = response[0][0] if response else []
			results[id_] = chose_word(query,answer)
	return results

if __name__ == '__main__':
	docs = util.collect_docs('../development.json')
	qs   = util.collect_questions('../development.json')
	results = run_baseline(qs,docs)
	with open('../result.json','w') as f:
		json.dump(results,f)
