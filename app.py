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
import copy

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
handler = handlers.RotatingFileHandler('foo.log', maxBytes=(1048576*5), backupCount=7)
logger.addHandler(handler)

CHOOSING, FAQ, ORDERS, CATALOG, MAIN = range(5)

orders = {}
uquery = {}
context = ''

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
# def make_payment(chat_id, username, bot):
#     title = "Payment Example"
#     description = "Payment Example"
#     payload = "Custom-Payload"
#     # In order to get a provider_token see https://core.telegram.org/bots/payments#getting-a-token
#     provider_token = "381764678:TEST:5997"
#     start_parameter = "test-payment"
#     currency = "RUB"
#     prices = []

#     if username in orders:
#         for item in orders[username]:
#             if 'ListPrice' in data[item]:
#                 item_price = Decimal(data[item]['ListPrice'].split('$')[1])
#                 logger.warning('item_price "%s"', str(item_price))
#                 prices.append(LabeledPrice(data[item]['Title'], int(item_price*100)))
#                 # prices.append(LabeledPrice(data[item]['Title'], 100))

#     # optionally pass need_name=True, need_phone_number=True,
#     # need_email=True, need_shipping_address=True, is_flexible=True
#         bot.sendInvoice(chat_id, title, description, payload,
#                 provider_token, start_parameter, currency, prices)
#     else:
#         bot.send_message(chat_id, 'Your card is empty')

# show user cart
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
        state_temp = copy.deepcopy(uquery[username]['state'])
        state_temp['stop'] = state_temp['start']
        state_temp['start'] = state_temp['start']-5
        
        r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(uquery[username]['query'], state_temp)]})
        response = json.loads(r.json())

        if len(response[0]['items']) == 0:
            bot.send_message(query.message.chat_id, "The search is empty. Please change your query.", parse_mode=telegram.ParseMode.HTML)
            return

        uquery[username]['scores'] = response[1]
        uquery[username]['items'] = response[0]['items']
        uquery[username]['entropy'] = response[0]['entropy']
        uquery[username]['total'] = int(response[0]['total'])
        uquery[username]['state'] = response[2]

        showitem(bot, query.message.chat_id, username)    
    
    if 'next' in query.data:
        state_temp = copy.deepcopy(uquery[username]['state'])
        state_temp['start'] = state_temp['stop']
        state_temp['stop'] = state_temp['stop']+5
        
        r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(uquery[username]['query'], state_temp)]})
        response = json.loads(r.json())

        if len(response[0]['items']) == 0:
            bot.send_message(query.message.chat_id, "The search is empty. Please change your query.", parse_mode=telegram.ParseMode.HTML)
            return

        uquery[username]['scores'] = response[1]
        uquery[username]['items'] = response[0]['items']
        uquery[username]['entropy'] = response[0]['entropy']
        uquery[username]['total'] = int(response[0]['total'])
        uquery[username]['state'] = response[2]

        # showitem(bot, query.message.chat_id, username, response[0]['items'], response[0]['entropy'], response[2])
        showitem(bot, query.message.chat_id, username)    
    # if 'payment' in query.data:
    #     # chat_id = update.message.chat_id
    #     make_payment(query.message.chat_id, username, bot)
       
    # add item to user cart 
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
        print('details parts:', parts, len(uquery[username]['items']))
        
        cats = ['Title', 'Manufacturer', 'Model', 'ListPrice', 'Binding', 'Color',
             'Genre', 'Author', 'Brand', 'Size', 'Feature']
        txt = ""
        for cat in cats:
            if cat in uquery[username]['items'][int(parts[1])]:
                if cat == 'ListPrice':
                    txt += '<b>' + cat + '</b>' + ':$' + uquery[username]['items'][int(parts[1])][cat].split('$')[1] + "\n"
                else:
                    txt += '<b>' + cat + '</b>' + ':' + uquery[username]['items'][int(parts[1])][cat] + "\n"
                    
        # act = [[InlineKeyboardButton('Add to card',callback_data='tocard:'+parts[1])]]
        # reply_markup = InlineKeyboardMarkup(act)

        # bot.send_message(query.message.chat_id, txt, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)
        bot.send_message(query.message.chat_id, txt, parse_mode=telegram.ParseMode.HTML)#, reply_markup=reply_markup)
        return 

    # answer FAQ question
    if 'answer' in query.data:
        parts = query.data.split(":")
        keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        bot.send_message(query.message.chat_id, parts[1], reply_markup=reply_markup)

    # show FAQ answer (pressed by button)
    if 'quest:' in query.data:
        parts = query.data.split(":")
        bot.send_message(query.message.chat_id, list(faq_js.items())[int(parts[1])][1])        

    # show entire FAQ
    # if query.data == 'showfaq':
    #     print("inside showfaq")

    #     keyboard = []
    #     for i, (key, value) in enumerate(faq_js.items()):
    #         print(key)
    #         print(value)
    #         keyboard.append([InlineKeyboardButton(key, callback_data='quest:'+str(i))])

    #     reply_markup = InlineKeyboardMarkup(keyboard)
    #     bot.send_message(query.message.chat_id, 'Press on question to know the answer', reply_markup=reply_markup)
    #     return FAQ

    # # FAQ section was pressed            
    # if query.data == 'faq':
    #     keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
    #     reply_markup = InlineKeyboardMarkup(keyboard)
    #     bot.send_message(query.message.chat_id, 'Type your question or press a button to show entire FAQ', reply_markup=reply_markup)
    #     return FAQ

    # Catalogue section was pressed
    if query.data == 'catalogue':
        bot.send_message(query.message.chat_id, 'Please type the product title')
        return CATALOG

    # filter by entropy
    if ":" in query.data:
    # if "query" in query.data:

        # state = json.loads(query.data)
        # print(state)

        # if 'query' not in state:
            # print('Search query is omitted')
            # return


        # state preredat'
        parts = query.data.split(":")

        # in case when it's restart and the you proceed the query
        if username not in uquery:
            uquery[username] = {}
            uquery[username]['state'] = {}

        uquery[username]['state']['start'] = 0
        uquery[username]['state']['stop'] = 5

        uquery[username]['state'][parts[0]] = parts[1]
        print(query.data)

        # print('specify key:',parts[0], 'value:', parts[1], 'state:', uquery[username]['state'])

        # r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(state['query'], state)]})
        r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(uquery[username]['query'], uquery[username]['state'])]})
        response = json.loads(r.json())

        uquery[username]['scores'] = response[1]
        uquery[username]['items'] = response[0]['items']
        uquery[username]['entropy'] = response[0]['entropy']
        uquery[username]['total'] = int(response[0]['total'])
        uquery[username]['state'] = response[2]

        # showitem(bot, query.message.chat_id, username, response[0]['items'], response[0]['entropy'], response[2])
        showitem(bot, query.message.chat_id, username)

        # return CATALOG
        # uquery[username]['start'] = 0
        # uquery[username]['stop'] = 4
        # parts = query.data.split(":")
        # if parts[0] in field:
        #     if parts[1] != 'undef':
        #         args = []
        #         for idx in uquery[username]['filter'][parts[0]]:#uquery[username]['results_args']:
        #             if parts[0] in data[idx]:
        #                 if data[idx][parts[0]].lower() == parts[1].lower():
        #                     args.append(idx)
                
        #         uquery[username]['results_args'] = args
        #     showitem(bot, query.message.chat_id, username)          

def showitem(bot, chat_id, username):#, items, entropy, state):

    global context
    print("SHOWITEM")
    # print("items", items)
    # print("entropy", entropy)
    # print("state", state)

    query = uquery[username]['query']    
    # start = uquery[username]['start'] if 'start' in uquery[username] else 0
    # stop = uquery[username]['stop'] if 'stop' in uquery[username] else 4

    if 'state' in uquery[username]:
        print("state", uquery[username]['state'])

        if 'stop' in uquery[username]['state']:
            stop = uquery[username]['state']['stop']
        else:
            stop = 5

        if 'start' in uquery[username]['state']:
            start = uquery[username]['state']['start']
        else:
            start = 0
    else:
        start = 0
        stop = 5
    
    # start = state['start'] if 'start' in state else 0
    # stop = state['stop'] if 'stop' in state else 5
    
    items = uquery[username]['items']
    scores = uquery[username]['scores']
    entropy = uquery[username]['entropy']
    total = uquery[username]['total']

    print("total", total)
    print("items", items)
    print("entropy", entropy)
    print("scores", scores)
    print("start", start, 'stop', stop)
    
    # print('showitem: start:', start, 'stop:', stop, 'results_args:', len(results_args))

    # if len(uquery[username]['items']) == 0:
    if len(items) == 0:
        bot.send_message(chat_id, "The search is empty. Please change your query.")
        return

    # step = 1
    
    # if stop>=len(results_args):
    #     print('REDEFINE STOP')
    #     stop = len(results_args)-1

    # last_item_id = results_args[stop]
    
    # if stop<start:
    #     step = -1
    #     last_item_id = results_args[start]

    # keyboard = []
    # keyboard.append([])

    # print('stop:', stop, 'last_item_id:', last_item_id, results_args[stop])

    # for idx, item in enumerate(uquery[username]['items']):
    for idx, item in enumerate(items[:-1]):
        act = [[]]
        act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(idx)))
        # act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(idx)))
        # act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
        reply_markup = InlineKeyboardMarkup(act)

        title = item['Title']

        if 'ListPrice' in item:
            title += " - <b>" + item['ListPrice'].split('$')[1] + "$</b>"

        bot.send_message(chat_id, title, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)


    # add the last element

    title = items[-1]['Title']
    if 'ListPrice' in items[-1]:
        title += " - <b>" + items[-1]['ListPrice'].split('$')[1] + "$</b>"

    act = [[],[]]
    act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(stop-1)))

    if start!=0:
        act[1].append(InlineKeyboardButton('Previous', callback_data='previous'))

    if stop<=total:
        act[1].append(InlineKeyboardButton('Next', callback_data='next'))
    
    reply_markup = InlineKeyboardMarkup(act)
    bot.send_message(chat_id, title, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)

    
        # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

    # act = [[],[]]
    # act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(last_item_id)))
    # act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(last_item_id)))

    
    
    
    
    # if int(start) > 0:
    #     act[1].append(InlineKeyboardButton('Previous', callback_data='previous'))

    # if len(results_args)-1>stop:
    #     act[1].append(InlineKeyboardButton('Next', callback_data='next'))

    # reply_markup = InlineKeyboardMarkup(act)

    # titlel = data[last_item_id]['Title']

    # if 'ListPrice' in data[last_item_id]:
    #     titlel += " - <b>" + data[last_item_id]['ListPrice'].split('$')[1] + "$</b>"

    # bot.send_message(chat_id, titlel, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)

    # print(uquery[username]['entropy'])
    # print('entropy comes', state)
    # if len(uquery[username]['entropy'])!=0:
    if len(entropy)!=0:
        act = [[]]
        # for entropy_value in uquery[username]['entropy'][0][2][:3]:
        for entropy_value in entropy[0][2][:3]:
            act[0].append(InlineKeyboardButton(entropy_value[0], callback_data=uquery[username]['entropy'][0][1]+':'+entropy_value[0]))
            # state_temp = copy.deepcopy(state)
            # state_temp[entropy[0][1]] = entropy_value[0]
            
            # if 'query' in state_temp:
                # del state_temp['query']

            # print('add', state_temp)
            # act[0].append(InlineKeyboardButton(entropy_value[0], callback_data=json.dumps(state_temp)))
        
        reply_markup = InlineKeyboardMarkup(act)
        # bot.send_message(chat_id, 'To specify the search, please choose a '+uquery[username]['entropy'][0][1], 
        print('adding entropy buttons')
        bot.send_message(chat_id, 'To specify the search, please choose a '+entropy[0][1], 
            reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)
        print('buttons were added')

def help(bot, update):
    update.message.reply_text('Please type your request in plain text')

# def classify_man(bot, username, query, start, stop):
#     r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(query, start, stop)]})
#     response = json.loads(r.json())
#     uquery[username] = {}
#     #uquery[username]['query'] = text
#     uquery[username]['start'] = 0
#     uquery[username]['stop'] = 4
#     uquery[username]['scores'] = response[1][0]
#     uquery[username]['items'] = response[0]['items']
#     uquery[username]['entropy'] = response[0]['entropy'][0]

#     showitem(bot, update.message.chat.id, username)
#     return CATALOG

def classify(bot, update):
    username = update._effective_user.username
    query = update.message.text
    global context


    # if context != '':
    #     print('classify: context:', context)
    #     results_args = uquery[username]['results_args']
    #     args = [idx for idx in results_args if query.lower() in data[idx][context].lower()]
    #     print('classify: args:', len(args))
    #     uquery[username]['start'] = 0
    #     uquery[username]['stop'] = 4
        
    #     if len(args) != 0:
    #         context = ''
    #         uquery[username]['results_args'] = args
    #         showitem(bot, update.message.chat.id, username)
    #         return
            
#    r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(query, 0, 5)]})
    r = requests.post("http://0.0.0.0:5000/ecommerce_bot", json={'context':[(query, {})]})

    #r = requests.post("http://127.0.0.1:5000", json={'context':[(query, 0, 5)]})
    # r = requests.post("http://0.0.0.0:5000/rankingemb_model", json={'context':[query], 'start':start, 'stop':stop})
    # intent = json.loads(r.json())['intent']
    # print(json.loads(r.json()))

    # if intent == 'faq':
    #     update.message.reply_text(json.loads(r.json())['text'])
    #     keyboard = [[InlineKeyboardButton("Show entire FAQ", callback_data='showfaq')]]
    #     reply_markup = InlineKeyboardMarkup(keyboard)
    #     update.message.reply_text('Type your question or press a button to show entire FAQ', reply_markup=reply_markup)
        
    # if intent == 'payment':
    #     showcart(bot, update.message.chat.id, username)

    # if intent == 'catalog':

    response = json.loads(r.json())

    if username in uquery:
        del uquery[username]

    # args = json.loads(r.json())['results_args']
    # scores = json.loads(r.json())['scores']

# distance limit by 0.5
    # args = [item for item in args if scores[item]<0.5]

    uquery[username] = {}
    uquery[username]['query'] = query
    uquery[username]['scores'] = response[1]
    uquery[username]['items'] = response[0]['items']
    uquery[username]['entropy'] = response[0]['entropy']
    uquery[username]['total'] = int(response[0]['total'])
    uquery[username]['state'] = response[2]

    if len(response[0]['items']) == 0:
    	update.message.reply_text('Nothing was found. Please change the query')
    	return 
    
    showitem(bot, update.message.chat.id, username)
    # showitem(bot, update.message.chat.id, username, response[0]['items'], response[0]['entropy'], {'query': query})
    return CATALOG

def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)

def done(bot, update, user_data):
    return ConversationHandler.END

def main():

    with open('./config.json', 'r') as f:
        config = json.load(f)

    if 'proxy' in config:
        REQUEST_KWARGS={ 'proxy_url': config['proxy'] } 
        updater = Updater(config['token'], request_kwargs=REQUEST_KWARGS)
        print("connecting proxy")
    else:
        updater = Updater(config['token'])
        print("no proxy")

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
    main()