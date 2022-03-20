from Classify import model_predict,load_model
from flask import Flask, render_template, request
app = Flask(__name__,template_folder='templates',static_folder="templates",static_url_path='')
model = load_model()

@app.route('/predict',methods=['GET','POST'])
def predict():
    isHead = False
    if request.method == 'POST':
        sent1 = request.form.get('sentence1')
        sent2 = request.form.get('sentence2')
        sentence = sent1 +"。"+ sent2
        response,result = model_predict(model,sentence)
    return render_template('index.html',isHead=isHead,sent1=sent1,sent2=sent2,response=response,result=result)

@app.route('/',methods=['GET','POST'])
def index():
    isHead = True
    return render_template('index.html',isHead=isHead)

if __name__ == '__main__':
    app.run()


