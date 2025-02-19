#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template_string, request, url_for, jsonify, redirect
import os
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


columns = ['simple', 'nationalism', 'reinforce beliefs', 'persuasive lang', 
           'emotional_img', 'patriotic_img', 'action_img', 'eastern front', 'southern front', 'western front']


# In[3]:


#functions to initialize all values
def initialize_df():
    df = pd.read_csv("final_df.csv")
    df['displayed']=0
    df['like'] = 0
    df['dislike'] = 0
    df['clicked']=0
    return df
def initialize_all_images():
    all_images = []
    for _, row in df.iterrows():
        all_images.append({'filename': row['img_ref'], 
                          'score': [row[col] for col in columns], 
                          'text': row['sequence']})
    return all_images

def initialize_neutr_table():
    neutr = pd.DataFrame(columns = columns)
    for i in range(len(columns)):
        neutr.at[0, columns[i]] = neutrality_score_try[i]
    return neutr
    


# In[4]:


#functions
##euclidean distance
def euclidean_distance(list1, dataframe, dfrow):
    return math.sqrt((list1[0] - dataframe.loc[dfrow,'simple']) ** 2 
                     + (list1[1] - dataframe.loc[dfrow,'nationalism']) ** 2
                     + (list1[2] - dataframe.loc[dfrow,'reinforce beliefs']) ** 2
                     + (list1[3] - dataframe.loc[dfrow,'persuasive lang']) ** 2
                     + (list1[4] - dataframe.loc[dfrow,'emotional_img']) ** 2
                     + (list1[5] - dataframe.loc[dfrow,'patriotic_img']) ** 2
                     + (list1[6] - dataframe.loc[dfrow,'action_img']) ** 2
                     + (list1[7] - dataframe.loc[dfrow,'eastern front']) ** 2
                     + (list1[8] - dataframe.loc[dfrow,'southern front']) ** 2
                     + (list1[9] - dataframe.loc[dfrow,'western front']) ** 2)
##sample values
def select_values(neutrality_score, covmatrix, number_of_posts):
    random_values = np.random.multivariate_normal(neutrality_score, covmatrix, number_of_posts)
    return random_values
##choose the closest posts
def closest_posts(random_values, posts_dataframe):
    all_closest_images = {}
    all_closest_rows ={}
    posts_dataframe['displayed'] = 0
    for i in range(len(random_values)):
        min_distance = float('inf')
        closest = None
        closest_row = None
        for row in range(len(posts_dataframe)):
            dist = euclidean_distance(random_values[i], posts_dataframe, row)
            if dist < min_distance and posts_dataframe.loc[row,'displayed']==0 and posts_dataframe.loc[row,'like']==0 and posts_dataframe.loc[row,'dislike']==0 and posts_dataframe.loc[row,'clicked']==0:
                min_distance = dist
                closest = posts_dataframe.loc[row, 'img_ref']
                closest_row = row
        all_closest_images[closest] = min_distance
        all_closest_rows[closest_row] = min_distance
        posts_dataframe.at[closest_row, 'displayed'] = 1
    return all_closest_rows


##choose posts
def select_posts(neutrality_score, covmatrix, number_of_posts, posts_dataframe):
    val = select_values(neutrality_score, covmatrix, number_of_posts)
    result = closest_posts(val, posts_dataframe)
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1]))
    sorted_list = list(sorted_result.keys())
    return sorted_list


# In[5]:


#functions for analysis at end of game

def load_data():
    return df

def generate_svg():
    data = load_data().sort_values(by="clicked_time")
    width, height = 500, 500
    min_x, max_x = -5, 2
    min_y, max_y = -3, 3

    def scale_x(x):
        return ((x - min_x) / (max_x - min_x)) * width

    def scale_y(y):
        return height - ((y - min_y) / (max_y - min_y)) * height

    svg_elements = ["<rect width='500' height='500' fill='#F5EFE0' />"]
    points = []

    for _, row in data.iterrows():
        x, y, click = scale_x(row['component 1']), scale_y(row['component 2']), row['clicked_time']
        points.append((x, y, click))

        # Determine color
        if row['like'] == row['dislike']:
            color = 'grey'
        elif row['like'] == 0 and row['dislike'] == 1:
            color = 'red'
        elif row['like'] == 1 and row['dislike'] == 0:
            color = 'green'

        svg_elements.append(f"<circle cx='{x}' cy='{y}' r='5' fill='{color}' stroke='black' />")

    points.sort(key=lambda p: p[2])  # Sort by click order
    clicked_points = [p for p in points if p[2] != 0]

    line_js_data = []  # Store line data for JS animation

    if clicked_points:
        start_x, start_y = width / 2, 0
        line_js_data.append(f"{{x1: {start_x}, y1: {start_y}, x2: {clicked_points[0][0]}, y2: {clicked_points[0][1]}}}")

        for i in range(len(clicked_points) - 1):
            x1, y1 = clicked_points[i][:2]
            x2, y2 = clicked_points[i + 1][:2]
            line_js_data.append(f"{{x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}}}")

    return "".join(svg_elements), line_js_data

def generate_plot2():
    filename = f"static/icons/plot2_{int(time.time())}.png"
    fig, ax = plt.subplots()
    neutr.plot(ax=ax)
    fig.set_facecolor('#F5EFE0')  # Change entire figure background
    ax.set_facecolor('#F5EFE0')  # Change only the plotting area
    ax.legend(title='Propaganda indicator', bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='#F5EFE0', edgecolor='black', framealpha=1)
    ax.grid(True)
    fig.savefig(filename, bbox_inches='tight', facecolor='#F5EFE0', transparent=True)
    # Close the figure to free memory
    plt.close(fig)
    # Return the filename for reference
    return filename


# In[7]:


app = Flask(__name__)

df = initialize_df()
neutrality_score_try =[np.mean(df[col]) for col in columns]
neutr = initialize_neutr_table()
all_images = initialize_all_images()
sd = [np.std(df[col]) for col in columns]
B = np.diag(sd)
counter = 0
df['displayed'] = 0
df['like'] = 0
df['dislike'] = 0
df['clicked'] = 0
df['clicked_time'] = 0
# Initialize the neutrality score at 0.5
neutrality_score = neutrality_score_try

# create lists to store liked and disliked posts
liked_posts = set()
disliked_posts = set()
# List of available images in the static folder
folder_path = "static"
# Get a list of all file names in the folder
available_images = df['img_ref'].tolist()

#start page template
start_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Start Page</title>
    <link href="https://fonts.googleapis.com/css2?family=Baskervville&display=swap" rel="stylesheet">
    <style>
    body {
        display: grid;
        place-items: center;
        font-family: 'Baskervville', serif;
        background-color: #F5EFE0;
    
    }
    .start-button {
         margin-top: 30px;
            padding: 15px 30px;
            font-family: inherit;
            font-size: 30px;
            background-color: #E8DEC9;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            }
    @keyframes shrink {
        0% {transform: scale(1); }
        100% {transform: scale(0.8); }
        }
    .shrink-image {
        width: 500px;
        height: auto;
        animation: shrink 2s ease-in-out forwards;
        animation-delay: 0.3s;
    }
    </style>
</head>

<body>
    
    <img src="{{ url_for('static', filename='icons/rabbitlogo-removebg.png') }}" alt="Logo" class="shrink-image" >
    <h2 style="text-align:center;"> Welcome to Rabbit, an interactive environment <br> where you can experience World War 1 information brought into the modern day. <br>
    Click start to be brought into the social media simulation. </h2>
    <h3 style="text-align:center;"> Disclaimer: all content is based on historical World War 1 documents but is AI generated. <br> 
    Do not make historical conclusions from the information presented. </h3>
     <form action="/home" method="get">
        <button type="submit" class="start-button">Start</button>
    </form>
</body>
</html>

"""

# Main page HTML template with clickable images and image scores
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Flask Button Example</title>
     <style>
        body {
            background-color: #F5EFE0;
            padding-top: 10px;
            overflow-y: auto; 
        }

        .banner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px; /* Adjust height as needed */
            background-color: #E8DEC9; /* A slightly darker shade for contrast */
            display: flex;
            padding: 10px 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
             z-index: 1000;
        }

    ]
        .banner img {
            height: 80px;
            width: auto;
            object-fit: contain;
        }



        
        .image-text-box {
            text-align: center;
            height: 530px;
            width: 400px;
            background-color: #E8DEC9;
            padding: 25px;
            font-size: 24px;
            font-family: 'Baskervville', serif;
            margin-top: 100px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 15px;
            position: relative;
        }

        .image-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr); /* 3 columns */
            grid-gap: 20px;
            margin-top: 100px; 
            justify-content: center; /* Centers horizontally */
            align-items: center; /* Aligns items in the middle */
            gap: 20px; /* Adds spacing between images */
            margin-top: 200px; /* Adds some space from the top */
        
        }
        .image-container img {
            width: 400px;
            height: auto;
            display: block;
            border-radius: 15px;
        }
        .image-wrapper {
            position:relative;
        }
    .like-btn {
        position: absolute;
        bottom: 0px; /* Position the button at the bottom */
        left: 0px; /* Position the button to the left */
        background: none; /* Remove background color */
        border: none; /* Remove border */
        padding: 0; /* Remove padding */
        margin: 0; /* Remove margin (if needed) */
        cursor: pointer; /* Change cursor to pointer on hover */
    }
    .dislike-btn {
        position: absolute;
        bottom: -3px; /* Position the button at the bottom */
        left: 50px; /* Position the button to the left */
        background: none; /* Remove background color */
        border: none; /* Remove border */
        padding: 0; /* Remove padding */
        margin: 0; /* Remove margin (if needed) */
        cursor: pointer; /* Change cursor to pointer on hover */
    }
    @keyframes buttonClick {
        0% {transform: scale(1);}
        50% {transform: scale(1.3);}
        100% {transform: scale(1);}
    }
   .like-btn.clicked, .dislike-btn.clicked {
    animation: buttonClick 0.3s ease-in-out;
}
    .exit-button {
        position: absolute;
        right: 60px;
        background:none;
        border: none;
        padding: 5px;
        cursor: pointer;
    }

    .exit-button img {
        height:60px;
        width:auto;
    }
</style>
        
   
<script>
function handleLike(postId) {
    fetch("/like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const likeButton = document.querySelector(`button[data-post-id='${postId}'][data-button-type='like']`);
            likeButton.classList.add("clicked");  // Add the "clicked" class
            document.getElementById("neutrality-score").innerText = data.neutrality_score.toFixed(1);

           setTimeout(() => {
                likeButton.classList.remove("clicked");
            }, 300);  // 300ms is the duration of the animation
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error("Error:", error));
}


function handleDislike(postId) {
    fetch("/dislike", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const dislikeButton = document.querySelector(`button[data-post-id='${postId}'][data-button-type='dislike']`);
            dislikeButton.classList.add("clicked");  // Add the "clicked" class
            document.getElementById("neutrality-score").innerText = data.neutrality_score.toFixed(1);

            // Remove the "clicked" class after the animation ends
            dislikeButton.addEventListener("animationend", () => {
                dislikeButton.classList.remove("clicked");
            }, { once: true });
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error("Error:", error));
}



document.addEventListener("DOMContentLoaded", function () {
    truncateText(".image-text-box h3", 50);  // Targets the h3 element inside the box
});

function truncateText(selector, maxLength) {
    document.querySelectorAll(selector).forEach(element => {
        let originalText = element.textContent.trim();
        if (originalText.length > maxLength) {
            element.textContent = originalText.slice(0, maxLength) + "...";
        }
    });
}

</script>
   
</head>

<body>
        
    <div class="banner">
        <img src="{{ url_for('static', filename='icons/rabbitlogo-removebg.png') }}" alt="Logo" >
           <form action="/exit" method="get">
        <button type="submit" class="exit-button"> 
            <img src="{{ url_for('static', filename='icons/exit.png') }}" alt="Click to exit game"}}>
        </button>
        </form>
    </div>


    <div class="image-container">

        <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[0].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
                <div class="image-text-box">
                <h3> {{ images[0].text}}</h3>
                <img src="{{ url_for('static', filename=images[0].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[0].filename }}" >
                </div>
            </button>
        </form>
   
    
        <button class="like-btn" data-post-id="{{ images[0].filename }}" data-button-type="like" onclick="handleLike('{{ images[0].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;">
         </button>
         <button class="dislike-btn" data-post-id="{{ images[0].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[0].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>
        
        <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[1].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
            <div class="image-text-box">
                <h3> {{ images[1].text}}</h3>
                <img src="{{ url_for('static', filename=images[1].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[1].filename }}" >
            </div>
            </button>
        </form>
        <button class="like-btn" data-post-id="{{ images[1].filename }}" data-button-type="like" onclick="handleLike('{{ images[1].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;" >
        </button>
        <button class="dislike-btn" data-post-id="{{ images[1].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[1].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>

        <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[2].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
            <div class="image-text-box">
                <h3> {{ images[2].text}}</h3>
                <img src="{{ url_for('static', filename=images[2].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[2].filename }}" >
            </div>
            </button>
        </form>
        <button class="like-btn" data-post-id="{{ images[2].filename }}" data-button-type="like" onclick="handleLike('{{ images[2].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;" >
        </button>
        <button class="dislike-btn" data-post-id="{{ images[2].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[2].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>

         <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[3].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
            <div class="image-text-box">
                <h3> {{ images[3].text}}</h3>
                <img src="{{ url_for('static', filename=images[3].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[3].filename }}" >
            </div>
            </button>
        </form>
        <button class="like-btn" data-post-id="{{ images[3].filename }}" data-button-type="like" onclick="handleLike('{{ images[3].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;" >
        </button>
        <button class="dislike-btn" data-post-id="{{ images[3].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[3].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>

        <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[4].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
            <div class="image-text-box">
                <h3> {{ images[4].text}}</h3>
                <img src="{{ url_for('static', filename=images[4].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[4].filename }}" >
            </div>
            </button>
        </form>
        <button class="like-btn" data-post-id="{{ images[4].filename }}" data-button-type="like" onclick="handleLike('{{ images[4].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;" >
        </button>
        <button class="dislike-btn" data-post-id="{{ images[4].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[4].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>

        <div class="image-wrapper">
        <form method="POST" action="/button-clicked/{{ images[5].filename }}" style="display: inline;">
            <button type="submit" style="border: none; background: none; padding: 0;">
            <div class="image-text-box">
                <h3> {{ images[5].text}}</h3>
                <img src="{{ url_for('static', filename=images[5].filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Click to see {{ images[5].filename }}" >
            </div>
            </button>
        </form>
        <button class="like-btn" data-post-id="{{ images[5].filename }}" data-button-type="like" onclick="handleLike('{{ images[5].filename }}')">
            <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;" >
        </button>
        <button class="dislike-btn" data-post-id="{{ images[5].filename }}" data-button-type="dislike" onclick="handleDislike('{{ images[5].filename }}')">
            <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;" >
        </button>
        </div>

    </div>

    

</body>

</html>
"""


# Template to display the clicked image
button_clicked_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Button Clicked</title>
    <style>
        body {
            background-color: #F5EFE0;
            padding-top: 5px;
        }

        .banner {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px; /* Adjust height as needed */
            background-color: #E8DEC9; /* A slightly darker shade for contrast */
            display: flex;
            padding: 10px 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }

    .home-button {
        background: none;  
        border: none;              
        font-size: 30px;         
        color: black;             
        cursor: pointer;          
        padding: none;               
        display: inline;           
        }   

        .home-button img {
            height: 80px;
            width: auto;
            object-fit: contain;             
        }
        
        
        
        .image-text-box {
            display: flex;        
            flex-direction: column; 
            text-overflow: ellipsis;
            justify-content: center;
            text-align: center;
            height:530px;
            width:400px;
            background-color: #E8DEC9;
            padding: 25px;
            font-size: 24px;
            font-family: 'Baskervville', serif;
            margin-top: 100px; 
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
            border-radius: 15px;
        }

        .image-container {
            display: flex;
            justify-content: center; /* Centers horizontally */
            align-items: center; /* Aligns items in the middle */
            gap: 20px; /* Adds spacing between images */
            margin-top: 200px; /* Adds some space from the top */
        
        }
        .image-container img {
            width: 400px;
            height: auto;
            display: block;
            border-radius: 15px;
        }
        .image-wrapper {
            position:relative;
        }

        
    .like-btn {
        position: absolute;
        bottom: 0px; /* Position the button at the bottom */
        left: 0px; /* Position the button to the left */
        background: none; /* Remove background color */
        border: none; /* Remove border */
        padding: 0; /* Remove padding */
        margin: 0; /* Remove margin (if needed) */
        cursor: pointer; /* Change cursor to pointer on hover */
    }
    .dislike-btn {
        position: absolute;
        bottom: -3px; /* Position the button at the bottom */
        left: 50px; /* Position the button to the left */
        background: none; /* Remove background color */
        border: none; /* Remove border */
        padding: 0; /* Remove padding */
        margin: 0; /* Remove margin (if needed) */
        cursor: pointer; /* Change cursor to pointer on hover */
    }
    @keyframes buttonClick {
    0% { transform: scale(1); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
}

/* Apply animation */
.like-btn.clicked, .dislike-btn.clicked {
    animation: buttonClick 0.3s ease-in-out;
}



</style>
        
<script>
function handleLike(postId) {
    fetch("/like", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const likeButton = document.querySelector(`button[data-post-id='${postId}'][data-button-type='like']`);
            likeButton.classList.add("clicked");  // Add the "clicked" class
            document.getElementById("neutrality-score").innerText = data.neutrality_score.toFixed(1);

           setTimeout(() => {
                likeButton.classList.remove("clicked");
            }, 300);  // 300ms is the duration of the animation
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error("Error:", error));
}


function handleDislike(postId) {
    fetch("/dislike", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ post_id: postId })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const dislikeButton = document.querySelector(`button[data-post-id='${postId}'][data-button-type='dislike']`);
            dislikeButton.classList.add("clicked");  // Add the "clicked" class
            document.getElementById("neutrality-score").innerText = data.neutrality_score.toFixed(1);

            // Remove the "clicked" class after the animation ends
            dislikeButton.addEventListener("animationend", () => {
                dislikeButton.classList.remove("clicked");
            }, { once: true });
        } else {
            alert(data.message);
        }
    })
    .catch(error => console.error("Error:", error));
}
</script>
   
</head>

<body>
    <div class="banner">
           <form action="/home" method="get">
        <button type="submit" class="home-button"> 
            <img src="{{ url_for('static', filename='icons/rabbitlogo-removebg.png') }}" alt="Click to go to home page"}}>
        </button>
    </form>
    </div>

    
 <div class="image-container">
        <div class="image-wrapper">
            <div class="image-text-box">
            <img src="{{ url_for('static', filename=image_filename) }}" style="border:3px solid black; height:400px; width:400px;" alt="Clicked Image">
            </div>

            <button class="like-btn" data-post-id="{{ image_filename }}" data-button-type="like" onclick="handleLike('{{ image_filename }}')">
                <img src="{{ url_for('static', filename='icons/happyrabbit.png') }}" alt="Like" style="width: 100px; height: auto;">
            </button>
            <button class="dislike-btn" data-post-id="{{ image_filename }}" data-button-type="dislike" onclick="handleDislike('{{ image_filename }}')">
                <img src="{{ url_for('static', filename='icons/sadrabbit.png') }}" alt="Dislike" style="width: 100px; height: auto;">
            </button> 
        </div>

        <div class="image-text-box">
            <p>{{ text }}</p>
        </div>
        
        </div>

    
</body>
</html>
"""

#Results Page

result_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flask PCA Animation</title>
        <style>
            body {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #F5EFE0;
                font-family: 'Baskervville', serif;
            }}
            h1 {{
                margin-bottom: 20px;
                
            }}
    
            .banner {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px; /* Adjust height as needed */
            background-color: #E8DEC9; /* A slightly darker shade for contrast */
            display: flex;
            justify-content: center; /* Centers horizontally */
            align-items: center; /* 
            padding: 10px 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}

    ]
        .banner h2 {{
            height: 80px;
            width: auto;
            object-fit: contain;
        }}


        </style>
    </head>

    <body>
        <div class="banner">
        <h2></h2>
        </div> 
        
        <div style="text-align: center;">
            <h1>Congratulations, you survived the war! </h1>
            <text style="font-size: 18pt">
            During this game you experienced several styles of reporting about the war.
            <br>
            Some of them were more neutral, while others were more propagandistic. 
            <br>
            Propagandistic messages are characterized by persuasive language, simplifying information and omitting nuance. <br>
            Instead of challenging your beliefs, they are reinforced. <br>
            Lastly, they play on your emotions and use a lot of we-vs-them speech.  <br>
            Similar to current social media, the algorithm of Rabbithole is constructed in such a way <br>
            that you see articles similar to the articles you liked before.  <br>
            Curious how far you spiraled into a rabbithole of propagandistic messages? Let's take a look! <br>
            </text>
        </div>

        
        <form method="POST" action="/result-clicked">
        <button type="submit" style="background-color: #E8DEC9;
            font-family: 'Baskervville', serif;
            font-size: 18pt;
            padding:10px;
            margin:15px;
            border-radius:15px;" >Your result</button>
        </form>
        
        

    </body>
    </html>
    """


@app.route("/result-clicked", methods=["POST"])
def result_clicked():
    svg_content, line_js_data = generate_svg()
    js_lines = "[" + ", ".join(line_js_data) + "]"

    result2_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Flask PCA Animation</title>
        <style>
            body {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #F5EFE0;
                font-family: 'Baskervville', serif;
            }}
            h1 {{
                margin-bottom: 20px;
                
            }}
           svg {{
                margin-top: 100px; /* Adjust this value to fit your needs */
            }}
            #rabbit {{
                position: absolute;
                width: 50px; /* Adjust the size of the rabbit image */
                height: auto;
                pointer-events: none;
            }}
            .banner {{
            display: flex;
            position:fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 80px; /* Adjust height as needed */
            justify-content: center; /* Pushes content apart */
            align-items: center; /* Aligns elements vertically */
            background-color: #E8DEC9;
            padding: 10px 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }}

    ]
             .banner h2 {{
            height: 80px;
            width: auto;
            object-fit: contain;
        }}
            .right-button {{
                position:absolute;
                background-color: #E8DEC9;
                font-family: 'Baskervville', serif;
                font-size: 14pt;
                right:60px;
                top:25px;
                padding: 10px;
                margin-left: auto; /* Pushes button to the right */
                border-radius: 15px;
                cursor: pointer;
            }}

        
        </style>
    </head>
    <body>

        <div class="banner">
        <h2>How similar are the posts that you interacted with?</h2>
        <form method="POST" action="/final-result">
        <button type="submit" class="right-button">Show Next Result</button>
        </form>
        </div> 
        
        <svg id="pca-svg" width="480" height="480" style="border: 3px solid black; border-radius:15px;">
            {svg_content}
        </svg>
        <p style="text-align:center; background-color:#E8DEC9;border: 3px solid black; border-radius:15px;padding:10px;">Follow the rabbit as it shows the path you took the posts. The dots represent all available posts <br>
         and are arranged by similarity, meaning that the more similar two posts are in regards to neutrality and topic, <br> the
          closer together they will be on the image. The dots are colored red and green <br> to represent which posts you disliked (red) and liked (green). </p>

        <img id="rabbit" src="static/icons/rabbit_jump.png" alt="rabbit jump" />
    
        <script>
            const lines = {js_lines};
            const svg = document.getElementById("pca-svg");
            const rabbit = document.getElementById("rabbit");

            function animateLine(line, x1, y1, x2, y2, duration, onComplete) {{
                let startTime = null;
                
                function step(timestamp) {{
                    if (!startTime) startTime = timestamp;
                    let progress = (timestamp - startTime) / duration;
                    if (progress > 1) progress = 1;

                    line.setAttribute("x2", x1 + (x2 - x1) * progress);
                    line.setAttribute("y2", y1 + (y2 - y1) * progress);

                    // Move the rabbit image
                    const newX = x1+480 + (x2 - x1) * progress - rabbit.width / 2;
                    const newY = y1+90 + (y2 - y1) * progress - rabbit.height / 2;
                    rabbit.style.left = newX + "px";
                    rabbit.style.top = newY + "px";

                    if (progress < 1) {{
                        requestAnimationFrame(step);
                    }} else if (onComplete) {{
                        onComplete();
                    }}
                }}

                requestAnimationFrame(step);
            }}

            function drawLinesSequentially(index) {{
                if (index >= lines.length) return;

                let data = lines[index];
                let line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", data.x1);
                line.setAttribute("y1", data.y1);
                line.setAttribute("x2", data.x1);
                line.setAttribute("y2", data.y1);
                line.setAttribute("stroke", "black");
                line.setAttribute("stroke-width", "2");
                svg.appendChild(line);

                animateLine(line, data.x1, data.y1, data.x2, data.y2, 800, () => drawLinesSequentially(index + 1));
            }}

            drawLinesSequentially(0);
        </script>
    </body>
    </html>
    """
    return render_template_string(result2_template) 



@app.route("/", methods=["GET"])
def start():
    neutrality_score_try =[np.mean(df[col]) for col in columns]
    neutr = initialize_neutr_table()
    counter = 0
    df['displayed'] = 0
    df['like'] = 0
    df['dislike'] = 0
    df['clicked'] = 0
    df['clicked_time'] = 0
    return render_template_string(start_template)


@app.route("/home", methods=["GET"])
def home():
    global neutrality_score
    res = select_posts(neutrality_score,B,6,df)
    selected_images = [all_images[res[i]] for i in range(len(res))]
    if len(res)<3:
        return "Error; Not enough posts selected", 500
    try:
        for image in selected_images:
            img = Image.open(f"static/{image}")
    except:
        res = select_posts(neutrality_score,B,6,df)
        selected_images = [all_images[res[i]] for i in range(len(res))]
    return render_template_string(html_template, neutrality_score=neutrality_score, images=selected_images)
@app.route('/exit', methods=["GET"])
def exit():
    return render_template_string(result_template)

@app.route("/final-result", methods=["POST"])
def line_plot():
    np.random.seed(42)
    columns = ['simple', 'nationalism', 'reinforce beliefs', 'persuasive lang', 
               'emotional_img', 'patriotic_img', 'action_img', 'eastern front', 
               'southern front', 'western front']
    #df = pd.DataFrame(np.random.uniform(-1, 1, size=(10, 10)), columns=columns)
    df = neutr

    # SVG dimensions
    width, height = 500, 300
    padding = 20
    legend_offset_x = width + 20  # Move the legend to the right of the graph

    # Normalize data to fit within SVG height
    y_max, y_min = 1, -1

    horizontal_offset = 100  # You can adjust this value to your preference

# Adjust the width of the SVG to accommodate the horizontal offset
    width_with_offset = width + horizontal_offset

    # Function to scale x-coordinates based on the offset
    def scale_x(j):
        return j * (width / (len(df) - 1)) + horizontal_offset
    def scale_y(value):
        return height - padding - ((value - y_min) / (y_max - y_min) * (height - 2 * padding))
    # Define colors for the lines
    colors = ["#FF0000", "#0000FF", "#00FF00", "#FF00FF", "#FFA500", "#800080", "#8B4513", "#000000"]

    # Generate points for each column
    def make_lines(columns_for_plot):
        polylines = []
        legends = []
        for i, column in enumerate(columns_for_plot):
            points = " ".join(f"{scale_x(j)},{scale_y(value)}" for j, value in enumerate(df[column]))
            polylines.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')

            # Legend positioning with the horizontal offset
            legend_y = 20 + (i * 20)  # Space out legend items
            legends.append(f'<rect x="{legend_offset_x+ horizontal_offset}" y="{legend_y - 10}" width="15" height="10" fill="{colors[i]}"/>')
            legends.append(f'<text x="{legend_offset_x + horizontal_offset+20}" y="{legend_y}" font-size="12" fill="black">{column}</text>')
        return polylines, legends

    # Create lines and legends for propaganda and topic data
    polylines_propaganda, legends_propaganda = make_lines(['simple', 'nationalism', 'reinforce beliefs', 'persuasive lang', 
                                                           'emotional_img', 'patriotic_img', 'action_img'])
    polylines_topic, legends_topic = make_lines(['eastern front', 'southern front', 'western front'])

    # Function to add x and y axes with ticks and labels
    def add_axes():
        axes = []

        # Draw x-axis (from 0 to len(df)-1), adjusting the x-coordinates
        axes.append(f'<line x1="{horizontal_offset}" y1="{height - padding}" x2="{width_with_offset}" y2="{height - padding}" stroke="black"/>')
        
        # Draw y-axis (from -1 to 1)
        axes.append(f'<line x1="{horizontal_offset}" y1="{padding}" x2="{horizontal_offset }" y2="{height - padding}" stroke="black"/>')

        # Add x-axis ticks and labels
        for i in range(len(df)):
            x = scale_x(i)
            axes.append(f'<line x1="{x}" y1="{height - padding + 5}" x2="{x}" y2="{height - padding - 5}" stroke="black"/>')

        # Add y-axis ticks and labels
        y_ticks = np.linspace(-1, 1, num=3)
        y_labels = ['neutral', 'average', 'propagandistic']
        for y_tick, label in zip(y_ticks, y_labels):
            y_pos = scale_y(y_tick)
            axes.append(f'<line x1="{horizontal_offset  - 5}" y1="{y_pos}" x2="{horizontal_offset  + 5}" y2="{y_pos}" stroke="black"/>')
            axes.append(f'<text x="{horizontal_offset  - 35}" y="{y_pos + 5}" font-size="12" fill="black" text-anchor="middle">{label}</text>')

        return axes

    axes = add_axes()
    html_result_2 = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Line Plot with Legend</title>
    </head>
    <style>
        body {{
            background-color: #F5EFE0;
            padding-top: 10px;
            overflow-y: auto; 
        }}
    .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
            justify-content:center;
            margin-top: 100px
        }}

        .image-wrapper {{
            display: flex;
            flex-direction: column;
            align-items: center;
            border: none;
            padding:10px; 
        }}

        .image-wrapper img {{
            width: 300px;
            height: auto;
            margin-bottom: 10px; 
            border-radius: 10px; 
        }}

        .text-box {{
            width: 810px;
            height: auto;
            padding: 10px;
            background: #E8DEC9;
            border: 3px solid black;
            border-radius: 15px;
    
        }}
         .banner {{
            position: fixed;
            display: flex; 
            justify-content: center; 
            align-items: center; 
            text-align: center; 
            top: 0;
            left: 0;
            width: 100%;
            height: 80px; /* Adjust height as needed */
            background-color: #E8DEC9; /* A slightly darker shade for contrast */
            display: flex;
            padding: 10px 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
             z-index: 1000;
        }}
         .banner h2 {{
            height: 80px;
            width: auto;
            object-fit: contain;
        }}


    svg {{
        background-color: #E8DEC9; margin-bottom: 15px; /* Change background color */
    }}
    </style>

    <body>
        <div class="banner">
        <h2> Let's break down your propaganda scores </h2>
        </div>
        <div class="container">
        <div class="text-box">
        <p style="text-align:center;"> Here, you can see how you scored for neutrality during the game! Each line represents 
        a different feature that measures propaganda, and you can see how your score for these 
        propaganda measures change with each like and dislike. </p>
        </div>
        <div class="image-wrapper">
        <svg width="{width_with_offset + 200}" height="{height}" style="border:3px solid black; border-radius:15px; padding:15px;">
            {''.join(axes)}
            {''.join(polylines_propaganda)}
            {''.join(legends_propaganda)}
        </svg>
        <svg width="{width_with_offset + 200}" height="{height}" style="border:3px solid black; border-radius:15px; padding:15px;">
            {''.join(axes)}
            {''.join(polylines_topic)}
            {''.join(legends_topic)}
        </svg>
        </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_result_2)


@app.route("/button-clicked/<image_filename>", methods=["POST"])
def button_clicked(image_filename):
    global neutrality_score, counter
    res = select_posts(neutrality_score, B,6,df)
    #selected_images = []
    for elem in res:
        filename = df.loc[elem, "img_ref"]
        text = df.loc[elem, "sequence"]
        image_score = [df.loc[elem, col] for col in columns]
        #df.loc[elem,"clicked"] = 1
        #df.loc[elem,"clicked_time"] = time.time()

    matching_row = df[df['img_ref'] == image_filename]
    

    if not matching_row.empty:
        image_score = matching_row.loc[matching_row.index[0], columns].tolist()
        text = matching_row.loc[matching_row.index[0], "sequence"]
        #matching_row['clicked'] = 1
        #matching_row["clicked_time"] = time.time()
    else:
        return "Error: Image not found", 404  # Handle the case where the image isn't found
    counter = sum(df['like'] + df['dislike'])

    if counter > 5:
        return redirect("/exit")

    return render_template_string(button_clicked_template, 
                                  neutrality_score=neutrality_score, 
                                  image_filename=image_filename, 
                                  image_score=image_score,
                                  text=text)


@app.route("/like", methods=["POST"])

def like():
    global neutrality_score
    data = request.get_json()
    image_filename = data.get('post_id')
    
    if image_filename in disliked_posts:
        return jsonify({"success": False, "message": "You cannot like a post you've disliked"})
    
    liked_posts.add(image_filename)

    matching_row = df[df['img_ref'] == image_filename].index
    df.loc[matching_row,"like"] = 1
    df.loc[matching_row,"clicked_time"] = time.time()
    matching_row_int = matching_row[0]  # Extract integer from index
    matching_row_neutr = all_images[matching_row_int]['score']
    neutrality_score = [min(1, score + 0.1 * matching_row_neutr[i]) if matching_row_neutr[i] > 0 else max(-1, score - 0.1 * matching_row_neutr[i]) for i, score in enumerate(neutrality_score)]
    #neutrality_score = [min(1, score + 0.1 * sd[i]) if score > 0 else min(1, score - 0.1 * sd[i]) for i, score in enumerate(neutrality_score)]
    #neutrality_score = [min(1.0, score + 0.1) if score > 0.5 else min(1.0, score - 0.1) for score in neutrality_score]
    #counter = sum(df['like']+df['dislike'])
    counter = len(neutr)

    for i in range(len(columns)):
        #neutr.at[len(neutr),columns[i]] = neutrality_score_try[i]
        
        neutr.at[counter, columns[i]] = neutrality_score[i]
    if counter > 5:
        return redirect("/exit")
    
    rendered_html = render_template_string(button_clicked_template, neutrality_score=neutrality_score, image_filename=image_filename)
    return jsonify({"success": True, "neutrality_score": neutrality_score, "html": rendered_html})



@app.route('/dislike', methods=['POST'])
def dislike():
    global neutrality_score
    data = request.get_json()
    image_filename = data.get('post_id')
    
    if image_filename in liked_posts:
        return jsonify({"success": False, "message": "You cannot dislike a post you've liked"})
    
    disliked_posts.add(image_filename)

    matching_row = df[df['img_ref'] == image_filename].index
    df.loc[matching_row,"dislike"] = 1
    df.loc[matching_row,"clicked_time"] = time.time()
    matching_row_int = matching_row[0]  # Extract integer from index
    matching_row_neutr = all_images[matching_row_int]['score']
    neutrality_score = [max(-1, score - 0.1 * matching_row_neutr[i]) if matching_row_neutr[i] > 0 else min(1, score + 0.1 * matching_row_neutr[i]) for i, score in enumerate(neutrality_score)]
    #neutrality_score = [max(0.0, score - 0.1) if score > 0.5 else max(0.0, score + 0.1) for score in neutrality_score]
    #counter = sum(df['like']+df['dislike'])
    counter = len(neutr)
    for i in range(len(columns)):
        #neutr.at[len(neutr),columns[i]] = neutrality_score_try[i]
        
        neutr.at[counter, columns[i]] = neutrality_score[i]
    if counter > 5:
        return redirect("/exit")
    
    rendered_html = render_template_string(button_clicked_template, neutrality_score=neutrality_score, image_filename=image_filename)
    return jsonify({"success": True, "neutrality_score": neutrality_score, "html": rendered_html})


# Run the Flask app
from werkzeug.serving import run_simple

if __name__ == "__main__":
    run_simple("localhost", 5001, app)


# In[19]:


neutr
       

