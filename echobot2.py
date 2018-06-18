#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple Bot to reply to Telegram messages.

This program is dedicated to the public domain under the CC0 license.

This Bot uses the Updater class to handle the bot.

First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.

Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import logging
from utils import *
from scipy.spatial.distance import cosine, euclidean
import numpy as np
import json
import pickle
import spacy
from spacy.tokens import Doc

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments bot and
# update. Error handlers also receive the raised TelegramError object in error.
def start(bot, update):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')


def help(bot, update):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')


def echo(bot, update):
    """Echo the user message."""
    text = update.message.text
    text_emb = emb.transform([nlp(text)], debug = True)[0]

    results_cos = []
    for idx, x_emb in enumerate(data_emb):

        if np.sum(x_emb) == 0:
            print('SKIP')
            continue
  
    # if y[idx] not in results_cos:
      # results_cos[y[idx]] = []
        
        results_cos.append([data_or[idx], cosine(text_emb, x_emb)])

    results_cos = sorted(results_cos,key=lambda x: x[1])

    update.message.reply_text(results_cos[0][0]['Title'])
        

    # return jsonify(results_cos[0][0]['Title'])
    # update.message.reply_text(update.message.text)
    # update.message.reply_text(update.message.text)

def error(bot, update, error):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, error)


def main():
    """Start the bot."""
    # Create the EventHandler and pass it your bot's token.
    updater = Updater("619576158:AAG5mkS442XJ_RFhNEZhSC-m-AgovYUawhU")

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_md', parser=False)
    data_or = load_data("./Amazon-E-commerce-Data-set/Data-sets/amazondata_Home_32865 668.txt")
    print("data loaded")

    with open("processed.pickle", "rb") as handle:
        doc_bytes, vocab_bytes = pickle.load(handle)
        print('pickle was loaded')

    nlp.vocab.from_bytes(vocab_bytes)
    docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
    print(len(docs))

    # docs_text = [doc.text for doc in docs]

    emb = MeanEmbeddingVectorizerSpacy()
    emb.fit(docs)
    data_emb = emb.transform(docs)
    print('tfidfed')

    main()
