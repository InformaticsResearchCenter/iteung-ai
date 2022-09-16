# -*- coding: utf-8 -*-
"""ITeung

# Preprocessing
"""

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import io
import os
import re
import requests
import csv
import datetime
import numpy as np
import pandas as pd
import random
import pickle

factory = StemmerFactory()
stemmer = factory.create_stemmer()

punct_re_escape = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
unknowns = ["gak paham","kurang ngerti","I don't know"]

list_indonesia_slang = pd.read_csv('./dataset/daftar-slang-bahasa-indonesia.csv', header=None).to_numpy()

data_slang = {}
for key, value in list_indonesia_slang:
    data_slang[key] = value

def dynamic_switcher(dict_data, key):
    return dict_data.get(key, None)

def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input

def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower())
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
  sentence = sentence.replace('teung', '')
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
  sentence = ' '.join(sentence.split())
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence)
  return sentence

df = pd.read_csv('./dataset/qa.csv', sep='|',usecols= ['question','answer'])
df.head()

question_length = {}
answer_length = {}

for index, row in df.iterrows():
  question = normalize_sentence(row['question'])
  question = normalize_sentence(question)
  question = stemmer.stem(question)

  if question_length.get(len(question.split())):
    question_length[len(question.split())] += 1
  else:
    question_length[len(question.split())] = 1

  if answer_length.get(len(str(row['answer']).split())):
    answer_length[len(str(row['answer']).split())] += 1
  else:
    answer_length[len(str(row['answer']).split())] = 1

question_length

answer_length

val_question_length = list(question_length.values())
key_question_length = list(question_length.keys())
key_val_question_length = list(zip(key_question_length, val_question_length))
df_question_length = pd.DataFrame(key_val_question_length, columns=['length_data', 'total_sentences'])
df_question_length.sort_values(by=['length_data'], inplace=True)
df_question_length.describe()

val_answer_length = list(answer_length.values())
key_answer_length = list(answer_length.keys())
key_val_answer_length = list(zip(key_answer_length, val_answer_length))
df_answer_length = pd.DataFrame(key_val_answer_length, columns=['length_data', 'total_sentences'])
df_answer_length.sort_values(by=['length_data'], inplace=True)
df_answer_length.describe()

data_length = 0

#filename = open('./dataset/clean_qa.txt', 'a+')
filename= './dataset/clean_qa.txt'
with open(filename, 'w', encoding='utf-8') as f:
  for index, row in df.iterrows():
    question = normalize_sentence(str(row['question']))
    question = normalize_sentence(question)
    question = stemmer.stem(question)

    answer = str(row['answer']).lower().replace('iteung', 'aku').replace('\n', ' ')

    if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
      body="{"+question+"}|<START> {"+answer+"} <END>"
      print(body, file=f)
      #filename.write(f"{question}\t<START> {answer} <END>\n")

#filename.close()
