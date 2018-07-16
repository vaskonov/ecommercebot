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
from decimal import Decimal
import requests

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
handler = handlers.RotatingFileHandler('foo.log', maxBytes=(1048576*5), backupCount=7)
logger.addHandler(handler)

# emb_mean = MeanEmbeddingVectorizerSpacy()
# emb_tfidf = TfidfEmbeddingVectorizerSpacy()

CHOOSING, FAQ, ORDERS, CATALOG, MAIN = range(5)

orders = {}
uquery = {}
field = ['Size', 'Brand', 'Author', 'Color', 'Genre']

def card(bot, update, args):
    update.message.reply_text('card was pressed')
    print(update)  
    print(args)

def start(bot, update):
    logger.warning('New start is detected "%s"', update)

    keyboard = [[InlineKeyboardButton("Catalog", callback_data='catalogue'),
                 InlineKeyboardButton("Orders", callback_data='opencard'),
                 InlineKeyboardButton("FAQ", callback_data='faq')]]

    reply_markup = InlineKeyboardMarkup(keyboard)

    # update.message.reply_text('Hi '+update._effective_user.first_name+'.Please choose one of the action. Or type your request in plain text.', reply_markup=reply_markup)
    
    update.message.reply_text('Hey '+update._effective_user.first_name, reply_markup=telegram.ReplyKeyboardRemove())
    update.message.reply_text('I am new e-commerce bot. I will help you to find products that you are looking for. Please choose one of the action. Or type your request in plain text.', reply_markup=reply_markup)
    return MAIN

#def start(bot, update):
#    update.message.reply_text('Please type the product query')
def make_payment(chat_id, username, bot):
    title = "Payment Example"
    description = "Payment Example"
    payload = "Custom-Payload"
    # In order to get a provider_token see https://core.telegram.org/bots/payments#getting-a-token
    provider_token = "381764678:TEST:5997"
    start_parameter = "test-payment"
    currency = "RUB"
    prices = []

    if username in orders:
        for item in orders[username]:
            if 'ListPrice' in data[item]:
                item_price = Decimal(data[item]['ListPrice'].split('$')[1])
                logger.warning('item_price "%s"', str(item_price))
                prices.append(LabeledPrice(data[item]['Title'], int(item_price*100)))
                # prices.append(LabeledPrice(data[item]['Title'], 100))

    # optionally pass need_name=True, need_phone_number=True,
    # need_email=True, need_shipping_address=True, is_flexible=True
        bot.sendInvoice(chat_id, title, description, payload,
                provider_token, start_parameter, currency, prices)
    else:
        bot.send_message(chat_id, 'Your card is empty')

def showcart(bot, chat_id, username):
#    username = update._effective_user.username
#    query = update.message.text

    if username in orders:
        for item_id in orders[username][0:-1]:
            bot.send_message(query.message.chat_id, data[item_id]['Title'])

        keyboard = [[InlineKeyboardButton("Make payment", callback_data='payment')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(chat_id, data[orders[username][-1]]['Title'], reply_markup=reply_markup)
#        update.message.reply_text(data[orders[username][-1]]['Title'], reply_markup=reply_markup)
    else:
        bot.send_message(chat_id, 'Your card is empty')
 #       update.message.reply_text('Your card is empty')

def button(bot, update):
    query = update.callback_query
    username = query.message.chat.username

    logger.warning('Button pressed "%s" - "%s"', query.data, query)

    if type(query.data) == int:
        update.message.reply_text(str(data[query.data]))

    if query.data == 'opencard':
        showcart(bot, query.message.chat_id, username)
        
    if 'below' in query.data:
        zerokeys = ['above', 'between', 'start', 'stop']
        uquery[username] = {k: v for k,v in uquery[username].items() if k not in zerokeys}
        uquery[username]['below'] = float(parts[1])
        showitem(bot, query.message.chat_id, username)

    if 'previous' in query.data:
        uquery[username]['stop'] = uquery[username]['start']-1
        uquery[username]['start'] = uquery[username]['start']-5
        showitem(bot, query.message.chat_id, username)

    if 'next' in query.data:
        uquery[username]['start'] = uquery[username]['stop']+1
        uquery[username]['stop'] = uquery[username]['stop']+5
        showitem(bot, query.message.chat_id, username)
    
    if 'payment' in query.data:
        # chat_id = update.message.chat_id
        make_payment(query.message.chat_id, username, bot)
        
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
        
        cats = ['Title', 'Manufacturer', 'Model', 'ListPrice', 'Binding', 'Color',
             'Genre', 'Author', 'Brand', 'Size', 'Feature']
        txt = ""
        for cat in cats:
            if cat in data[int(parts[1])]:
                if cat == 'ListPrice':
                    txt += '<b>' + cat + '</b>' + ':$' + data[int(parts[1])][cat].split('$')[1] + "\n"
                else:
                    txt += '<b>' + cat + '</b>' + ':' + data[int(parts[1])][cat] + "\n"
                    
        act = [[InlineKeyboardButton('Add to card',callback_data='tocard:'+parts[1])]]
        reply_markup = InlineKeyboardMarkup(act)

        bot.send_message(query.message.chat_id, txt, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)

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
        return FAQ
            
    if query.data == 'faq':
        keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, 'Type your question or press a button to show entire FAQ', reply_markup=reply_markup)
        return FAQ

    if query.data == 'catalogue':
        bot.send_message(query.message.chat_id, 'Please type the product title')
        return CATALOG

    if ":" in query.data:
        uquery[username]['start'] = 0
        uquery[username]['stop'] = 4
        parts = query.data.split(":")
        if parts[0] in field:
            if parts[1] != 'undef':
                args = []
                for idx in uquery[username]['results_args']:
                    if parts[0] in data[idx]:
                        if data[idx][parts[0]].lower() == parts[1].lower():
                            args.append(idx)
                
                uquery[username]['results_args'] = args
            showitem(bot, query.message.chat_id, username)          

    if query.data == '1':
#      update.message.reply_text('Please type the product query.')
      bot.edit_message_text(text="Please type the product query.", chat_id=query.message.chat_id, message_id=query.message.message_id)

    if query.data == '2':
      #update.message.reply_text('You do not have orders.') 
      bot.edit_message_text(text="You do not have orders.", chat_id=query.message.chat_id, message_id=query.message.message_id)

    if query.data == '3':
   #   update.message.reply_text('Please type your question.')
      bot.edit_message_text(text="Please type your question.", chat_id=query.message.chat_id, message_id=query.message.message_id)



def showitem(bot, chat_id, username):

    from scipy.stats import entropy
    from collections import Counter

    fil = {'Size':[], 'Brand':[], 'Author':[], 'Color':[], 'Genre':[]}

    print("showitem")
    #query = uquery[username]['query']
    start = uquery[username]['start'] if 'start' in uquery[username] else 0
    stop = uquery[username]['stop'] if 'stop' in uquery[username] else 4
    results_args = uquery[username]['results_args']
    scores = uquery[username]['scores']

    print('showitem: start:', start, 'stop:', stop, 'results_args:', len(results_args))

    if len(results_args) == 0:
        bot.send_message(chat_id, "The search is empty. Please change your query.")
        return

   # logger.warning('Next was pressed "%s"', str(uquery[username]))
    # results_args, scores = search(query)
    # results_args, scores = search(query)

    step = 1
    
    if stop>=len(results_args):
        print('REDEFINE STOP')
        stop = len(results_args)-1

    # last_item_id = results_args[min(stop, len(results_args)-1)]
    last_item_id = results_args[stop]
    
    if stop<start:
        step = -1
        last_item_id = results_args[start]

    keyboard = []
    keyboard.append([])

    print('stop:', stop, 'last_item_id:', last_item_id, results_args[stop])

    for idx in results_args[start:start+20:step]:
        for key in field:
            if key in data[idx]:
                fil[key].append(data[idx][key].lower())
            # else:
                # fil[key].append('undef')
    print(fil)

    filcount = {}
    entropy_list = []

    for key, value in fil.items():
        filcount[key] = list(Counter(list(value)).values())
        print(key, filcount[key])
        entropy_list.append((key, entropy(filcount[key], base=2)))
    print(entropy_list)

    max_entropy = sorted(entropy_list, key=lambda x: x[1], reverse=True)[0]

    max_entropy_field = max_entropy[0]
    print(max_entropy_field)

    for idx in results_args[start:stop:step]:
        logger.warning('Result "%s" with score "%s"', str(data[idx]['Title']), str(scores[idx]))
        # update.message.reply_text(str(docs[idx].text))
        act = [[]]
        act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(idx)))
        act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
        reply_markup = InlineKeyboardMarkup(act)

        title = data[idx]['Title']

        if 'ListPrice' in data[idx]:
            title += " - <b>" + data[idx]['ListPrice'].split('$')[1] + "$</b>"

        bot.send_message(chat_id, title, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)
    
        # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

    act = [[],[]]
    act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(last_item_id)))
    act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(last_item_id)))

    if int(start) > 0:
        act[1].append(InlineKeyboardButton('Previous', callback_data='previous'))

    if len(results_args)-1>stop:
        act[1].append(InlineKeyboardButton('Next', callback_data='next'))

    reply_markup = InlineKeyboardMarkup(act)

    titlel = data[last_item_id]['Title']

    if 'ListPrice' in data[last_item_id]:
        titlel += " - <b>" + data[last_item_id]['ListPrice'].split('$')[1] + "$</b>"

    bot.send_message(chat_id, titlel, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)

    # here goes the check
    
    if max_entropy[1]>0.5:
        act = [[]]
        
        for val in Counter(fil[max_entropy_field]).most_common()[:4]:
            if val[0] != 'undef':
                act[0].append(InlineKeyboardButton(val[0], callback_data=max_entropy_field+':'+val[0]))

        reply_markup = InlineKeyboardMarkup(act)
        bot.send_message(chat_id, 'To specify the search, please choose a '+max_entropy_field.lower(), 
            reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)    

def help(bot, update):
    update.message.reply_text('Please type your request in plain text')

def classify(bot, update):
    username = update._effective_user.username
    query = update.message.text

    r = requests.post("http://0.0.0.0:5000/rankingemb_model", json={'context':[[query]]})
    # r = requests.post("http://0.0.0.0:5000/rankingemb_model", json={'context':[query], 'start':start, 'stop':stop})
    intent = json.loads(r.json())['intent']
    print(json.loads(r.json()))

    if intent == 'faq':
        update.message.reply_text(json.loads(r.json())['text'])
        keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Type your question or press a button to show entire FAQ', reply_markup=reply_markup)
        
    if intent == 'payment':
        showcart(bot, update.message.chat.id, username)

    if intent == 'catalog':
        if username in uquery:
            del uquery[username]

        args = json.loads(r.json())['results_args']
        scores = json.loads(r.json())['scores']

        args = [item for item in args if scores[item]<0.5]

        uquery[username] = {}
        #uquery[username]['query'] = text
        uquery[username]['start'] = 0
        uquery[username]['stop'] = 4
        uquery[username]['scores'] = scores
        uquery[username]['results_args'] = args

        if len(args) == 0:
            update.message.reply_text('Nothing was found. Please change the query')
            return 
        
        showitem(bot, update.message.chat.id, username)
        return CATALOG

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
            MAIN: [CallbackQueryHandler(button), MessageHandler(Filters.text, classify)],#main_process
            FAQ: [CallbackQueryHandler(button), MessageHandler(Filters.text, classify),],#[MessageHandler(Filters.text, faq_process),],
            CATALOG: [CallbackQueryHandler(button), MessageHandler(Filters.text, classify),], #[MessageHandler(Filters.text, catalogue_process),]
            # ORDERS: [MessageHandler(Filters.text, catalogue_process),]
        },
        fallbacks=[CommandHandler('start', start),],
        )
    dp.add_handler(CommandHandler("help", help))

    print("get command", Filters.text)
    dp.add_handler(conv_handler)

    # dp.add_handler(CommandHandler("start", start))
    # dp.add_handler(CommandHandler("card", card, pass_args=True))
    dp.add_handler(CallbackQueryHandler(button))
    #dp.add_handler(MessageHandler(Filters.text, main_process))
    dp.add_handler(MessageHandler(Filters.text, classify))

    # dp.add_handler(MessageHandler(Filters.text, echo))

    dp.add_error_handler(error)
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':

    with open('faq.json', 'r') as f:
        faq_js = json.load(f)
    
    with open('/tmp/phones.pickle', 'rb') as handle:
        data = pickle.load(handle)
        logger.warning('Data set loaded "%s"', str(len(data)))

    main()