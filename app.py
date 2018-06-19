#!/usr/bin/env python
# -*- coding: utf-8 -*-

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import telegram
import logging
from logging import handlers
from utils import *
from scipy.spatial.distance import cosine, euclidean
import numpy as np
import json
import pickle
import spacy
from spacy.tokens import Doc
import math

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
handler = handlers.RotatingFileHandler('foo.log', maxBytes=(1048576*5), backupCount=7)
logger.addHandler(handler)

def start(bot, update):
    keyboard = [[InlineKeyboardButton("Catalog", callback_data='1'),
                 InlineKeyboardButton("Orders", callback_data='2'),
                 InlineKeyboardButton("FAQ", callback_data='3')]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Please choose:', reply_markup=reply_markup)

#def start(bot, update):
#    update.message.reply_text('Please type the product query')

def button(bot, update):
    query = update.callback_query

    if query.data == '1':
#      update.message.reply_text('Please type the product query.')
      bot.edit_message_text(text="Please type the product query.", chat_id=query.message.chat_id, message_id=query.message.message_id)

    if query.data == '2':
      #update.message.reply_text('You do not have orders.') 
      bot.edit_message_text(text="You do not have orders.", chat_id=query.message.chat_id, message_id=query.message.message_id)

    if query.data == '3':
   #   update.message.reply_text('Please type your question.')
      bot.edit_message_text(text="Please type your question.", chat_id=query.message.chat_id, message_id=query.message.message_id)


def help(bot, update):
    update.message.reply_text('Please type the product query')


def echo(bot, update):
    text = update.message.text
    logger.warning('Incoming query "%s"', text)

    text_mean_emb = emb_mean.transform([nlp(text)])[0]
    text_tfidf_emb = emb_tfidf.transform([nlp(text)])[0]

    bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    results_mean = [cosine(text_mean_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_mean]
    results_tfidf = [cosine(text_tfidf_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_tfidf]


    # for idx, x_emb in enumerate(data_mean):

    #     mean_emb = data_mean[idx]
    #     tfidf_emb = data_tfidf[idx]
        
    #     if np.sum(mean_emb) == 0:
    #         print('SKIP')
    #         continue

    #     if np.sum(tfidf_emb) == 0:
    #         print('SKIP')
    #         continue

    #     scores = {}
    #     scores['mean_cosine'] = cosine(text_mean_emb, mean_emb)
    #     scores['tfidf_cosine'] = cosine(text_tfidf_emb, tfidf_emb)
        
    # results_cos.append([docs[idx], np.sum(list(scores.values())), scores])

    results = np.mean([results_mean,results_tfidf], axis=0)
    # results_cos = sorted(results_cos,key=lambda x: x[1])
    results_args = np.argsort(results)

    
    for idx in results_args[:5]:
        logger.warning('Result "%s" with scores "%s"', str(docs[idx].text), str(results[idx]))
        update.message.reply_text(str(docs[idx].text))

def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    updater = Updater("619576158:AAG5mkS442XJ_RFhNEZhSC-m-AgovYUawhU")

    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CallbackQueryHandler(button))
    dp.add_handler(CommandHandler("help", help))

    dp.add_handler(MessageHandler(Filters.text, echo))

    dp.add_error_handler(error)

    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    
    # nlp = spacy.load('en_core_web_md', parser=False)
    nlp = spacy.load('en_vectors_web_lg', parser=False)

    with open("processed.pickle.home.big", "rb") as handle:
    #with open("processed.pickle.big", "rb") as handle:
        doc_bytes, vocab_bytes = pickle.load(handle)
        print('pickle was loaded')

    nlp.vocab.from_bytes(vocab_bytes)
    docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
    print(len(docs))

    # docs_text = [doc.text for doc in docs]

    emb_mean = MeanEmbeddingVectorizerSpacy()
    emb_mean.fit(docs)
    data_mean = emb_mean.transform(docs)
    print('meaned')

    emb_tfidf = TfidfEmbeddingVectorizerSpacy()
    emb_tfidf.fit(docs)
    data_tfidf = emb_tfidf.transform(docs)    
    print('tfidfed')

    main()
