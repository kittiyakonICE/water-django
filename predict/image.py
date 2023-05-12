import sqlite3
from tkinter import *
from tkinter import filedialog



"""root = Tk()

def filedlog():
    global get_image
    get_image = filedialog.askopenfilenames(title="SELECT IMAGE",fileTypes=( ("png" , "*.png"),("jpg" , "*.jpg"),("Alifile","*.*")))

def convert_image_into_binary(filename):
    with open(filename,'rb')as file:
        photo_image = file.read()
    return photo_image


def insert_image():
    image_database = sqlite3.connect("Image_data.db")
    data = image_database.cursor()
    
    for image in get_image:
        insert_photo = convert_image_into_binary(image)
        data.execute("INSERT INTO Image Values(:image"),{'image': insert_photo}

        image_database.commit()
        image_database.close()


def create_database():
    image_database = sqlite3.connect*("image_data.db")
    data = image_database.cursor()
    
    data.execute("INSERT INTO Image Vale(:image)")
    
    image_database.commit()
    image_database.close()
    
create_database()

select_image = Button(root,text="Select Image")
select_image.grid(row=0,column=0, pady=(100,0), padx=100)

save_image = Button(root,text="Save",command=insert_image)
save_image.grid(row=1,column=0)

root.mainlopp()"""

    
    
    