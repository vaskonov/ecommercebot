#!/usr/bin/env python
# -*- coding: utf-8 -*-

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, ConversationHandler, RegexHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice, ShippingOption
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

nlp = spacy.load('en_vectors_web_lg', parser=False)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
handler = handlers.RotatingFileHandler('foo.log', maxBytes=(1048576*5), backupCount=7)
logger.addHandler(handler)

emb_mean = MeanEmbeddingVectorizerSpacy()
# emb_tfidf = TfidfEmbeddingVectorizerSpacy()

CHOOSING, FAQ, ORDERS, CATALOGUE, MAIN = range(5)
orders = {}

def card(bot, update, args):
    update.message.reply_text('card was pressed')
    print(update)  
    print(args)

def start(bot, update):
    logger.warning('New start is detected "%s"', update)

    keyboard = [[InlineKeyboardButton("Catalog", callback_data='catalogue'),
                 InlineKeyboardButton("Orders", callback_data='tocard'),
                 InlineKeyboardButton("FAQ", callback_data='faq')]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Hi '+update._effective_user.first_name+'.Please choose one of the action. Or type your request in free text.', reply_markup=reply_markup)
    return MAIN

#def start(bot, update):
#    update.message.reply_text('Please type the product query')

def button(bot, update):
    query = update.callback_query
    username = query.message.chat.username

    logger.warning('Button pressed "%s" - "%s"', query.data, query)

    if type(query.data) == int:
        update.message.reply_text(str(data[query.data]))

    if query.data == 'opencard':
        if username in orders:
            for item_id in orders[username][0:-1]:
                bot.send_message(query.message.chat_id, data[item_id]['Title'])

            keyboard = [[InlineKeyboardButton("Make a payment", callback_data='payment')]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            bot.send_message(query.message.chat_id, data[orders[username][-1]]['Title'], reply_markup=reply_markup)

    if 'payment' in query.data:
        # chat_id = update.message.chat_id
        title = "Payment Example"
        description = "Payment Example using python-telegram-bot"
        payload = "Custom-Payload"
        # In order to get a provider_token see https://core.telegram.org/bots/payments#getting-a-token
        provider_token = "PROVIDER_TOKEN"
        start_parameter = "test-payment"
        currency = "USD"
        prices = []
    
        if username in orders:
            for item in orders[username]:
                if 'ListPrice' in data[idx]:
                    item_price = float(data[idx]['ListPrice'].split('$')[1])
                    prices.append([LabeledPrice(data[idx]['Title'], item_price)])

        # optionally pass need_name=True, need_phone_number=True,
        # need_email=True, need_shipping_address=True, is_flexible=True
        bot.sendInvoice(query.message.chat_id, title, description, payload,
                    provider_token, start_parameter, currency, prices)

    if 'tocard' in query.data:
        parts = query.data.split(":")
        if username not in orders:
            orders[username] = []

        orders[username].append(int(parts[1]))
        keyboard = [[InlineKeyboardButton("Open card", callback_data='opencard')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, 'The item was added to card.', reply_markup=reply_markup)

    if 'details' in query.data:
        parts = query.data.split(":")
        cats = ['Title', 'Manufacturer', 'Model', 'ListPrice', 'Binding']
        for cat in cats:
            if cat in data[int(parts[1])]:
                txt = cat + ':' + data[int(parts[1])][cat]
                if cat == 'ListPrice':
                    txt = cat+':$'+data[int(parts[1])][cat].split('$')[1]
                bot.send_message(query.message.chat_id, txt)

    if 'answer' in query.data:
        parts = query.data.split(":")
        keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, parts[1], reply_markup=reply_markup)

    if 'quest:' in query.data:
        parts = query.data.split(":")
        bot.send_message(query.message.chat_id, list(faq_js.items())[int(parts[1])][1])        
        
    if query.data == 'showfaq':
        print("inside showfaq")

        keyboard = []
        for i, (key, value) in enumerate(faq_js.items()):
            print(key)
            print(value)
            keyboard.append([InlineKeyboardButton(key, callback_data='quest:'+str(i))])

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, 'Press on question to know the answer', reply_markup=reply_markup)        
            
    if query.data == 'faq':
        keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, 'Type your question or press a button to show entire FAQ', reply_markup=reply_markup)
        return FAQ

    if query.data == 'catalogue':
        bot.send_message(query.message.chat_id, 'Please type the product title')
        return CATALOGUE

        # custom_keyboard = [['FAQ']]
        # reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        # bot.send_message(chat_id=query.message.chat_id, text="Type your question or press FAQ", reply_markup=reply_markup)

    if 'showitems' in query.data:
        parts = query.data.split(':')
        start = int(parts[2])
        stop = int(parts[3])

        step = 1
        if stop<start:
            step = -1

        logger.warning('Next was pressed "%s"', query.data)
        results_args = search(parts[1])

        keyboard = []
        keyboard.append([])
    
        for idx in results_args[start:stop:step]:
            logger.warning('Result "%s" with scores', str(docs[idx].text))
            # update.message.reply_text(str(docs[idx].text))
            act = []
            act.append([InlineKeyboardButton('Show details', callback_data='details:'+str(idx))])
            act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
            reply_markup = InlineKeyboardMarkup(act)
            bot.send_message(query.message.chat_id, str(docs[idx].text), reply_markup=reply_markup)
        
            # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

        if int(parts[2]) != 0:
            keyboard[0].append(InlineKeyboardButton('Previous', callback_data='showitems:'+parts[1]+':'+str(start-4)+':'+str(start)))

        keyboard[0].append(InlineKeyboardButton('Next', callback_data='showitems:'+parts[1]+':'+str(stop)+':'+str(stop+4)))

        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, 'Please choose one of the items or press Next|Previous', reply_markup=reply_markup)
        # update.message.reply_text('Please choose one of the items or press Next:', reply_markup=reply_markup)

      # bot.edit_message_text(text=str(data[query.data]), chat_id=query.message.chat_id, message_id=query.message.message_id)

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

def classify(query, items):
    items_nlped = [emb_mean.transform([nlp(text)])[0] for text in items]
    query_nlped = emb_mean.transform([nlp(query)])[0]
    results = [cosine(query_nlped, emb) if np.sum(emb)!=0 else math.inf for emb in items_nlped]
    return results

def search(text):
    text_mean_emb = emb_mean.transform([nlp(text)])[0]
 #   text_tfidf_emb = emb_tfidf.transform([nlp(text)])[0]

    # bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
    # bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

    results_mean = [cosine(text_mean_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_mean]
#    results_tfidf = [cosine(text_tfidf_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_tfidf]
        #results = np.mean([results_mean,results_tfidf], axis=0)

    results = np.mean([results_mean], axis=0)
    # results_cos = sorted(results_cos,key=lambda x: x[1])
    results_args = np.argsort(results)
    return results_args

def faq_process(bot, update):
    query = update.message.text
    logger.warning('FAQ query "%s"', query)

    ques = list(faq_js.keys())
    ans = list(faq_js.values())

    results = classify(query, ques)
    
    results_arg = np.argsort(results)
    
    logger.warning('Results "%s" with threshold "%s"', str(ques[results_arg[0]]), str(results[results_arg[0]]))

    if results[results_arg[0]] < 0.2:
        update.message.reply_text(ans[results_arg[0]])
        # bot.edit_message_text(text=ques[results_arg[0]], chat_id=query.message.chat_id, message_id=query.message.message_id)
    else:
        update.message.reply_text('Please rephrase your question')
        # bot.edit_message_text(text='Please rephrase your question', chat_id=query.message.chat_id, message_id=query.message.message_id)


def main_process(bot, update):
    text = update.message.text
    logger.warning('MAIN process "%s"', text)

    ques = list(faq_js.keys())
    ans = list(faq_js.values())

    results = classify(query, ques)
    
    results_arg = np.argsort(results)
    
    logger.warning('Results "%s" with threshold "%s"', str(ques[results_arg[0]]), str(results[results_arg[0]]))

    if results[results_arg[0]] < 0.2:
        update.message.reply_text(ans[results_arg[0]])
        # bot.edit_message_text(text=ques[results_arg[0]], chat_id=query.message.chat_id, message_id=query.message.message_id)
    else:
        update.message.reply_text('This is not a question from FAQ')


def catalogue_process(bot, update):
    text = update.message.text

    

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


    results_args = search(text)
    
    for idx in results_args[:4]:
        logger.warning('Result "%s" with scores', str(docs[idx].text))

        act = []
        act.append([InlineKeyboardButton('Show details', callback_data='details:'+str(idx))])
        act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
        reply_markup = InlineKeyboardMarkup(act)
        update.message.reply_text(str(docs[idx].text), reply_markup=reply_markup)

        # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

    keyboard = []
    keyboard.append([InlineKeyboardButton('Next', callback_data='showitems:'+text+':'+'5:10')])

    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Select one of the items or select Next', reply_markup=reply_markup)

def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)

def done(bot, update, user_data):
    return ConversationHandler.END

def main():
    updater = Updater("619576158:AAG5mkS442XJ_RFhNEZhSC-m-AgovYUawhU")

    dp = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[
                        CommandHandler('start', start)],
        states = {
            MAIN: [CallbackQueryHandler(button), MessageHandler(Filters.text, main_process)],
            FAQ: [MessageHandler(Filters.text, faq_process),],
            CATALOGUE: [MessageHandler(Filters.text, catalogue_process),]
            # ORDERS: [MessageHandler(Filters.text, catalogue_process),]
        },
        fallbacks=[CommandHandler('start', start)],
        )
    dp.add_handler(CommandHandler("help", help))

    print("get command", Filters.text)
    dp.add_handler(conv_handler)

    # dp.add_handler(CommandHandler("start", start))
    # dp.add_handler(CommandHandler("card", card, pass_args=True))
    dp.add_handler(CallbackQueryHandler(button))
    # dp.add_handler(CommandHandler("help", help))

    # dp.add_handler(MessageHandler(Filters.text, echo))

    dp.add_error_handler(error)
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':

    with open('faq.json', 'r') as f:
        faq_js = json.load(f)
    
    # nlp = spacy.load('en_core_web_md', parser=False)

    with open('phones.pickle', 'rb') as handle:
        data = pickle.load(handle)
        logger.warning('Data set loaded "%s"', str(len(data)))

    with open("processed.pickle.phones.big", "rb") as handle:
    #with open("processed.pickle.big", "rb") as handle:
        doc_bytes, vocab_bytes = pickle.load(handle)

    nlp.vocab.from_bytes(vocab_bytes)
    docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
    logger.warning('Nlped data set loaded "%s"', str(len(docs)))

    # docs_text = [doc.text for doc in docs]

    emb_mean.fit(docs)
    data_mean = emb_mean.transform(docs)
    print('meaned')

#    emb_tfidf = TfidfEmbeddingVectorizerSpacy()
#    emb_tfidf.fit(docs)
#    data_tfidf = emb_tfidf.transform(docs)    
#    print('tfidfed')

    main()
