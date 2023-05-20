from datetime import datetime
import pickle
import cv2
import os
from flask import Flask,request,render_template,redirect, url_for, jsonify,flash
from datetime import date
# from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import csv
import keras
import keras.backend as k
from keras.layers import Conv2D,MaxPooling2D,SpatialDropout2D,Flatten,Dropout,Dense
from keras.models import Sequential,load_model
from keras.optimizers import Adam
import keras.utils as image



#### Defining Flask App
app = Flask(__name__)


#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")



#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(1)
except:
    cap = cv2.VideoCapture(0)


#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time,Temperature,Mask\n')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


#### extract the face from an image
def extract_faces(img):
    if img!=[]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

#### Identify face using ML model
def identify_face(facearray,face_recognition_model):
    face_recognition_model = joblib.load('static/face_recognition_model.pkl')
    
    return face_recognition_model.predict(facearray)



@app.route('/predict', methods=['GET'])
def predict():
    # UNCOMMENT THE FOLLOWING CODE TO TRAIN THE CNN FROM SCRATCH#
    # BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK#
    # model=Sequential()
    # model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
    # model.add(MaxPooling2D() )
    # model.add(Conv2D(32,(3,3),activation='relu'))
    # model.add(MaxPooling2D() )
    # model.add(Conv2D(32,(3,3),activation='relu'))
    # model.add(MaxPooling2D() )
    # model.add(Flatten())
    # model.add(Dense(100,activation='relu'))
    # model.add(Dense(1,activation='sigmoid'))

    # model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    # from keras.preprocessing.image import ImageDataGenerator
    # train_datagen = ImageDataGenerator(
    #         rescale=1./255,
    #         shear_range=0.2,
    #         zoom_range=0.2,
    #         horizontal_flip=True)

    # test_datagen = ImageDataGenerator(rescale=1./255)

    # training_set = train_datagen.flow_from_directory(
    #         'FaceMaskDetector/train',
    #         target_size=(150,150),
    #         batch_size=16 ,
    #         class_mode='binary')

    # test_set = test_datagen.flow_from_directory(
    #         'FaceMaskDetector/test',
    #         target_size=(150,150),
    #         batch_size=16,
    #         class_mode='binary')

    # model_saved=model.fit_generator(
    #         training_set,
    #         epochs=10,
    #         validation_data=test_set,

    #         )

    # model.save('mymodel.h5',model_saved)

    #To test for individual images

    # mymodel=load_model('mymodel.h5')
    # #test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
    # test_image=image.load_img(r'FaceMaskDetector/test/with_mask/1-with-mask.jpg',
    #                           target_size=(150,150,3))
    # test_image
    # test_image=image.img_to_array(test_image)
    # test_image=np.expand_dims(test_image,axis=0)
    # mymodel.predict(test_image)[0][0]




    # IMPLEMENTING LIVE DETECTION OF FACE MASK

    mymodel=load_model('mymodel.h5')

    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while cap.isOpened():
        _,img=cap.read()
        face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('temp.jpg',face_img)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict(test_image)[0][0]
            if pred==1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            datet=str(datetime.now())
            cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

        cv2.imshow('img',img)

        if cv2.waitKey(1)==ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    names,rolls,times,temps,masks,l = extract_attendance()  
    return render_template('home2.html',names=names,rolls=rolls,times=times,temps=temps,masks=masks,l=l,totalreg=totalreg(),datetoday2=datetoday2) 
    

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces') 
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    joblib.dump(knn,'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    temps = 36.5
    masks = df['Mask']
   
    l = len(df)
    return names,rolls,times,temps,masks,l


#### Add Attendance of a specific employee
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")
    temps = 36.5
  
    
  
    
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'{username},{userid},{current_time},{temps}\n')


## Routing Functions

#The main page
@app.route('/')
def home():
    names,rolls,times,temps,masks,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,temps=temps,masks=masks,l=l,totalreg=totalreg(),datetoday2=datetoday2) 


## This function will run when we click on the Click here button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')
    
    # Load face recognition model
    face_recognition_model = joblib.load('static/face_recognition_model.pkl', 'r')

    # Load face mask detection model
    face_mask_model = load_model('mymodel.h5')

    # Start video capture
    cap = cv2.VideoCapture(0)
    ret = True

    while ret:
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            # Resize the face image to match the input size of the face recognition model
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))

            # Perform face recognition on the face
            identified_person = identify_face(face.reshape(1, -1), face_recognition_model)[0]

            # Perform face mask detection on the face
            face_img = cv2.resize(frame[y:y+h, x:x+w], (150, 150))
            test_image = image.img_to_array(face_img)
            test_image = np.expand_dims(test_image, axis=0)
            pred = face_mask_model.predict(test_image)[0][0]

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)

            # Display name of the recognized person
            cv2.putText(frame, f'{identified_person}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 20), 2)

            # Display face mask detection result
            if pred == 1:
                cv2.putText(frame, 'NO MASK', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'MASK', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

         

            # Add the attendance record
            add_attendance(identified_person)

        # Display the frame
        cv2.imshow('Attendance', frame)

        # Quit the loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release video capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

    # Extract the attendance records and render the attendance page
    names, rolls, times, temps, masks, l = extract_attendance()
    return render_template('home2.html', names=names, rolls=rolls, times=times, temps=temps, masks=masks, l=l, totalreg=totalreg(), datetoday2=datetoday2)

#### This function will run when a new employee registers
@app.route('/add',methods=['GET','POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/'+newusername+'_'+str(newuserid)
    temps = 35.5
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    cap = cv2.VideoCapture(0)
    i,j = 0,0
    while 1:
        _,frame = cap.read()
        faces = extract_faces(frame)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame,f'Images Captured: {i}/50',(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 20),2,cv2.LINE_AA)
            if j%10==0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name,frame[y:y+h,x:x+w])
                i+=1
            j+=1
        if j==500:
            break
        cv2.imshow('Adding new User',frame)
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names,rolls,times,temps,masks,l = extract_attendance()    
    return render_template('home.html',names=names,rolls=rolls,times=times,temps=temps,masks=masks,l=l,totalreg=totalreg(),datetoday2=datetoday2) 

@app.route('/generateReport',methods=["GET","POST"])
def report():
    if request.method=="GET":
        generateReport()
        return render_template ("home.html",mess= 'Report generated successfully. Please check your email for the report.')
    else:
        return render_template('home2.html',mess= 'Please click on the Generate Report button to generate the report.')
def generateReport():
    
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication
    from datetime import date



    datetoday = date.today().strftime("%m_%d_%y")
    datetoday2 = date.today().strftime("%d-%B-%Y")

    # Email settings
    email_from = 'danmuendo4@gmail.com'
    email_password = 'iipzqqfgifozuctn'
    email_to = 'eugineogembo@gmail.com'

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = email_from
    msg['To'] = email_to
    msg['Subject'] = f'Attendance Report for {datetoday2}'

    # Add the attendance file as an attachment
    with open(f'Attendance/Attendance-{datetoday}.csv', 'rb') as f:
        attach = MIMEApplication(f.read(), _subtype='csv')
        attach.add_header('Content-Disposition', 'attachment', filename=f'Attendance-{datetoday}.csv')
        msg.attach(attach)

    # Send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login(email_from, email_password)
        smtp.send_message(msg)
   

#### create a redirect for home2.html
@app.route('/home2')
def home2():
    return render_template('home2.html')

#### This is main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
