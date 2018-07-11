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

# nlp = spacy.load('en_vectors_web_lg', parser=False)
# nlp = spacy.load('en_core_web_lg', parser=False)
# nlp_en = spacy.load('en', parser=False)

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
        cats = ['Title', 'Manufacturer', 'Model', 'ListPrice', 'Binding', 'Feature']
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

        # custom_keyboard = [['FAQ']]
        # reply_markup = telegram.ReplyKeyboardMarkup(custom_keyboard)
        # bot.send_message(chat_id=query.message.chat_id, text="Type your question or press FAQ", reply_markup=reply_markup)

    # if 'showitems' in query.data:
    #     parts = query.data.split(':')
    #     start = int(parts[2])
    #     stop = int(parts[3])

    #     logger.warning('Next was pressed "%s"', query.data)
    #     results_args, scores = search(parts[1])

    #     step = 1
    #     last_item_id = results_args[stop]

    #     if stop<start:
    #         step = -1
    #         last_item_id = results_args[start]

    #     keyboard = []
    #     keyboard.append([])
    
    #     for idx in results_args[start:stop:step]:
    #         logger.warning('Result "%s" with scores', str(docs[idx].text))
    #         # update.message.reply_text(str(docs[idx].text))
    #         act = [[]]
    #         act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(idx)))
    #         act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
    #         reply_markup = InlineKeyboardMarkup(act)
    #         bot.send_message(query.message.chat_id, str(docs[idx].text), reply_markup=reply_markup)
        
    #         # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

    #     act = [[],[]]
    #     act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(last_item_id)))
    #     act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(last_item_id)))
    
    #     if int(parts[2]) > 0:
    #         act[1].append(InlineKeyboardButton('Previous', callback_data='showitems:'+parts[1]+':'+str(start-5)+':'+str(start-1)))

    #     act[1].append(InlineKeyboardButton('Next', callback_data='showitems:'+parts[1]+':'+str(stop+1)+':'+str(stop+5)))

    #     reply_markup = InlineKeyboardMarkup(act)
    #     bot.send_message(query.message.chat_id, str(docs[last_item_id].text), reply_markup=reply_markup)
    #     # update.message.reply_text('Please choose one of the items or press Next:', reply_markup=reply_markup)

    #   # bot.edit_message_text(text=str(data[query.data]), chat_id=query.message.chat_id, message_id=query.message.message_id)

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

    print("showitem")
    #query = uquery[username]['query']
    start = uquery[username]['start'] if 'start' in uquery[username] else 0
    stop = uquery[username]['stop'] if 'stop' in uquery[username] else 4
    results_args = uquery[username]['results_args']
    scores = uquery[username]['scores']

    if len(scores) == 0:
        bot.send_message(chat_id, "The search is empty. Please change your query.")
        return

   # logger.warning('Next was pressed "%s"', str(uquery[username]))
    # results_args, scores = search(query)
    # results_args, scores = search(query)

    step = 1
    last_item_id = results_args[stop]

    if stop<start:
        step = -1
        last_item_id = results_args[start]

    keyboard = []
    keyboard.append([])

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

    act[1].append(InlineKeyboardButton('Next', callback_data='next'))

    reply_markup = InlineKeyboardMarkup(act)

    titlel = data[last_item_id]['Title']

    if 'ListPrice' in data[last_item_id]:
        titlel += " - <b>" + data[last_item_id]['ListPrice'].split('$')[1] + "$</b>"

    bot.send_message(chat_id, titlel, reply_markup=reply_markup, parse_mode=telegram.ParseMode.HTML)

def help(bot, update):
    update.message.reply_text('Please type your request in plain text')

# def classify_intent(query, intents):

#     query_nlped = emb_mean.transform([nlp(query)])[0]
#     intents_scores = []
#     for intent, sens in intents.items():
#         for sen in sens:
#             sen_nlped = emb_mean.transform([nlp(sen)])[0]

#             if np.sum(sen_nlped)!=0:
#                 score = cosine(query_nlped, sen_nlped)
#             else:
#                 score = math.inf
        
#             print(query, sen, score)
   
#             intents_scores.append((intent, score))

#     return [score for score in intents_scores if score[1]>=0]

# def classify(query, items):
#     items_nlped = [emb_mean.transform([nlp(text)])[0] for text in items]
#     query_nlped = emb_mean.transform([nlp(query)])[0]
#     results = [cosine(query_nlped, emb) if np.sum(emb)!=0 else math.inf for emb in items_nlped]
#     return results

# def search(text):
#     text_mean_emb = emb_mean.transform([nlp(text)])[0]
#  #   text_tfidf_emb = emb_tfidf.transform([nlp(text)])[0]

#     # bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)
#     # bot.send_chat_action(chat_id=update.message.chat_id, action=telegram.ChatAction.TYPING)

#     results_mean = [cosine(text_mean_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_mean]
# #    results_tfidf = [cosine(text_tfidf_emb, emb) if np.sum(emb)!=0 else math.inf for emb in data_tfidf]
#         #results = np.mean([results_mean,results_tfidf], axis=0)

#     scores = np.mean([results_mean], axis=0)
#     # results_cos = sorted(results_cos,key=lambda x: x[1])
#     results_args = np.argsort(scores)
#     return results_args, scores

# def faq_process(bot, update):
#     query = update.message.text
#     logger.warning('FAQ query "%s"', query)

#     ques = list(faq_js.keys())
#     ans = list(faq_js.values())

#     results = classify(query, ques)
    
#     results_arg = np.argsort(results)
    
#     logger.warning('Results "%s" with threshold "%s"', str(ques[results_arg[0]]), str(results[results_arg[0]]))

#     if results[results_arg[0]] < 0.2:
#         update.message.reply_text(ans[results_arg[0]])
#         # bot.edit_message_text(text=ques[results_arg[0]], chat_id=query.message.chat_id, message_id=query.message.message_id)
#     else:
#         update.message.reply_text('Please rephrase your question')
#         # bot.edit_message_text(text='Please rephrase your question', chat_id=query.message.chat_id, message_id=query.message.message_id)


# def main_process(bot, update):

#     catalogue_process(bot, update)
#     return True

#     query = update.message.text
#     logger.warning('MAIN process "%s"', query)

#     intents = {
#         'catalog': ['I am looking for', 'I want to buy', 'I need', 'I search', 'do you have'],
#         'payment': ['I want to pay', 'I need to pay for my order', 'please receive a payment']
#     }

#     ques = list(faq_js.keys())
#     ans = list(faq_js.values())

#     results = classify(query, ques)
    
#     results_arg = np.argsort(results)
    
#     logger.warning('Results "%s" with threshold "%s"', str(ques[results_arg[0]]), str(results[results_arg[0]]))

#     if results[results_arg[0]] < 0.2:
#         update.message.reply_text(ans[results_arg[0]])
#         # bot.edit_message_text(text=ques[results_arg[0]], chat_id=query.message.chat_id, message_id=query.message.message_id)
#     else:
#         logger.warning('This is not a question from FAQ')
#         doc = filter_nlp(nlp(query))
#         for w in doc:
#           print(w, w.tag_)
#         nns = [w.text for w in doc if w.tag_ in ['NNP', 'NN', 'JJ']]
#         sal_phrase = [w.text for w in doc if w.tag_ not in ['NNP', 'NN', 'JJ']]
#         logger.warning('parsing: salient phrase: "%s" nns: "%s"', sal_phrase, nns)

#         scores = sorted(classify_intent(" ".join(sal_phrase), intents), key=lambda x: x[1])
#         print(scores)

#         if scores[0][1] < 0.2:
#             if scores[0][0] == 'catalog':
#                 update.message.text = " ".join(nns)
#                 catalogue_process(bot, update)
#             if scores[0][0] == 'payment':
#                 make_payment(update.message.chat.id, update.message.chat.username, bot)
#         else:
#             update.message.reply_text('Please rephrase your request')

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

        uquery[username] = {}
        #uquery[username]['query'] = text
        uquery[username]['start'] = 0
        uquery[username]['stop'] = 4
        uquery[username]['scores'] = json.loads(r.json())['scores']
        uquery[username]['results_args'] = json.loads(r.json())['results_args']
        
        showitem(bot, update.message.chat.id, username)
        return CATALOG

    # text = update.message.text
    
    # uquery[username] = {}
    # uquery[username]['query'] = text
    # # uquery[username]['start'] = 5
    # # uquery[username]['stop'] = 10

    # print("catalogue_process")

    # # for idx, x_emb in enumerate(data_mean):

    # #     mean_emb = data_mean[idx]
    # #     tfidf_emb = data_tfidf[idx]
        
    # #     if np.sum(mean_emb) == 0:
    # #         print('SKIP')
    # #         continue

    # #     if np.sum(tfidf_emb) == 0:
    # #         print('SKIP')
    # #         continue

    # #     scores = {}
    # #     scores['mean_cosine'] = cosine(text_mean_emb, mean_emb)
    # #     scores['tfidf_cosine'] = cosine(text_tfidf_emb, tfidf_emb)
        
    # # results_cos.append([docs[idx], np.sum(list(scores.values())), scores])

    # # uquery

    # results_args, scores = search(text)
    # prices = sorted([float(data[idx]['ListPrice'].split('$')[1]) for idx in results_args if scores[idx]<0.4 and 'ListPrice' in data[idx]])

    # price_first = prices[round(float(len(prices))/3)]
    # price_last = prices[round(float(len(prices)*2)/3)]

    # logger.warning('Prices of high relevance "%s"', str(prices))
    # logger.warning('First: "%s" Last: "%s"', str(price_first), str(price_last))
    
    # for idx in results_args[:4]:
    #     logger.warning('Result "%s" with score "%s"', str(docs[idx].text), str(scores[idx]))

    #     act = []
    #     act.append([InlineKeyboardButton('Show details', callback_data='details:'+str(idx))])
    #     act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(idx)))
    #     reply_markup = InlineKeyboardMarkup(act)
    #     update.message.reply_text(str(docs[idx].text), reply_markup=reply_markup)

    #     # keyboard.append([InlineKeyboardButton(docs[idx].text, callback_data=idx)])

    # act = [[],[]]
    # act[0].append(InlineKeyboardButton('Show details', callback_data='details:'+str(results_args[4])))
    # act[0].append(InlineKeyboardButton('Add to card', callback_data='tocard:'+str(results_args[4])))
    # act[1].append(InlineKeyboardButton('Next', callback_data='showitems:'+text+':'+'5:10'))
    # reply_markup = InlineKeyboardMarkup(act)
    # update.message.reply_text(str(docs[results_args[4]].text), reply_markup=reply_markup)

    # pr = [[]]
    # pr[0].append(InlineKeyboardButton('Below '+str(price_first)+'$', callback_data='below:'+str(price_first)))
    # pr[0].append(InlineKeyboardButton('Between '+str(price_first)+'$ and '+str(price_last)+'$', callback_data='between:'+str(price_first)+":"+str(price_last)))
    # pr[0].append(InlineKeyboardButton('Above '+str(price_last)+'$', callback_data='above:'+str(price_last)))
    # reply_markup = InlineKeyboardMarkup(pr)
    # update.message.reply_text('Please select a price range', reply_markup=reply_markup)

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
    
    # nlp = spacy.load('en_core_web_md', parser=False)

    with open('/tmp/phones.pickle', 'rb') as handle:
        data = pickle.load(handle)
        logger.warning('Data set loaded "%s"', str(len(data)))

    # with open("processed.pickle.phones.big", "rb") as handle:
    #with open("processed.pickle.big", "rb") as handle:
        # doc_bytes, vocab_bytes = pickle.load(handle)

    # nlp.vocab.from_bytes(vocab_bytes)
    # docs = [Doc(nlp.vocab).from_bytes(b) for b in doc_bytes]
    # logger.warning('Nlped data set loaded "%s"', str(len(docs)))


    # emb_mean.fit(docs)
    # data_mean = emb_mean.transform(docs)
    # print('meaned')

#    emb_tfidf = TfidfEmbeddingVectorizerSpacy()
#    emb_tfidf.fit(docs)
#    data_tfidf = emb_tfidf.transform(docs)    
#    print('tfidfed')

    main()
