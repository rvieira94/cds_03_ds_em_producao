from flask import Response, request, Flask
import pandas as pd
import requests
import json
import os

TOKEN = ''

# bot status
#'https://api.telegram.org/botTOKEN/getMe'

# get new messages
#'https://api.telegram.org/botTOKEN/getUpdates'

# send message to chat
#'https://api.telegram.org/botTOKEN/sendMessage?chat_id=&text='

def send_message( chat_id, text ):
    url = 'https://api.telegram.org/bot{}/sendMessage?chat_id={}'.format(TOKEN, chat_id)
    r = requests.post( url, json={'text': text })
    print( 'send_message Status Code {}'.format(r.status_code))

    return None

def load_dataset( store_id ):
    # loading test dataset
    df10 = pd.read_csv('data/test.csv')
    df_store_raw = pd.read_csv('data/store.csv')

    # merge test + store datasets
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store')

    # choose store for prediction
    df_test = df_test[df_test['Store'] == store_id]

    if df_test.empty:
        data = 'error'
    else:
        # remove closed days
        df_test = df_test[(df_test['Open'] != 0) & (~df_test['Open'].isnull())]
        df_test = df_test.drop('Id', axis=1)

        # convert Dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))

    print( 'load_dataset done' )

    return data

def predict( data ):
    # API call
    url = 'https://p01dsp-model.herokuapp.com/rossmann/predict'
    header = {'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data=data, headers=header)
    print('predict Status code {}'.format(r.status_code))

    return pd.DataFrame(r.json(), columns=r.json()[0].keys())

def parse_message(message):

    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/','')

    try:
        store_id = int(store_id)

    except ValueError:
        store_id = 'error'

    print('parse_message done {} / {}'.format(chat_id,store_id))

    return chat_id, store_id

# API initialize
app = Flask( __name__ )
@app.route('/',methods=['GET','POST'])

def index():
    if request.method == 'POST':
        message = request.get_json()        

        #if len(message) > 1:
        #    send_message(message['message']['chat']['id'], 'Too many messages. Please send a message and wait for the response.')
        #    return Response('OK', status=200)
        #else:

        chat_id, store_id = parse_message(message)

        if store_id != 'error':
            # loading data
            data = load_dataset(store_id)
            if data != 'error':
                # prediction
                d1 = predict(data)
                # calculation
                d2 = d1[['store','prediction']].groupby('store').sum().reset_index()
                # send message
                msg = 'Store number {} will sell US${:,.2f} in the next 6 weeks'.format(
                    d2['store'].values[0], d2['prediction'].values[0])
                send_message(chat_id, msg)
                return Response('OK', status=200)
            else:
                send_message(chat_id, 'Store not available')
                return Response('OK', status=200)
        else:
            send_message(chat_id, 'Are you sure this is a store ID?')
            return Response('OK', status=200)
    else:
        send_message(chat_id, '<h1>Rossmann Telegram BOT</h1>')
        return Response('OK', status=200)

if __name__ == '__main__':
    port = os.environ.get('PORT',5000)
    app.run(host='0.0.0.0', port=port)