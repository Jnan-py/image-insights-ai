from dotenv import load_dotenv
import google.generativeai as genai 
import os
import streamlit as st
import re 
import requests
import json
from google.generativeai import types
from streamlit_option_menu import option_menu
from PIL import Image, ImageDraw, ImageFont, ImageColor
import random
import base64
from io import BytesIO
import numpy as np


load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

aud_mod = genai.GenerativeModel(
    model_name= "gemini-2.0-flash",
    system_instruction= "You are a good image analyzer through which you will be able to find the related audience with respect to the image"
   )

img_model = genai.GenerativeModel(
    model_name = "gemini-2.0-flash",
    system_instruction= "You are an image analyzer model, Analyze the image and provide insights",
)

obj_det_model_2d = genai.GenerativeModel(
    model_name = "gemini-2.0-flash",
    system_instruction="""
            Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
            If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
            """
)

obj_det_model_1d = genai.GenerativeModel(
    model_name = "gemini-2.0-flash",
    system_instruction="""
        Return points as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        The answer should follow the STRICT JSON format: [{"point": array , "label": string}, ...]. One element a line.
        """
)

def get_from_uploads(image_file):
    file_path = get_local_path(image_file.name)
    with open(file_path, "wb") as f:
        while chunk := image_file.read(1024 * 1024):  
            f.write(chunk)
    return file_path 

@st.cache_data
def load_img(img_url):
    filename = os.path.basename(img_url).split("?")[0]
    response = requests.head(img_url)
    content_type = response.headers.get('content-type', '')
    extension = content_type.split("/")[-1]
    if extension not in ['jpeg', 'png', 'gif', 'bmp', 'webp']:
        extension = 'jpg'

    file_path = get_local_path(f"{filename.split('.')[0]}.{extension}")

    if 'image' not in content_type:
        raise ValueError("URL does not point to an image")

    img_resp = requests.get(img_url, stream=True)
    img_resp.raise_for_status()

    with open(file_path, 'wb') as f:
            f.write(img_resp.content)

    return file_path

def get_local_path(filename):
    safe_filename = os.path.basename(filename).replace(" ", "_")
    downloads_folder = os.path.join("downloads")
    os.makedirs(downloads_folder, exist_ok=True)    
    return os.path.join(downloads_folder, safe_filename) 

def parse_gemini_response(response_text):
    try:
        cleaned = re.sub(r'```json|```', '', response_text)
        cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', cleaned)
        json_str = re.search(r'\{[\s\S]*\}', cleaned)
        if json_str:
            return json.loads(json_str.group())
        return None

    except Exception as e:
        st.error(f"Failed to parse response: {str(e)}")
        return None 

def get_about(file_path):
   prompt = """ Generate the related audience for whom the provided image will be useful and required, 
   And also classify the image as per the two options : "Educational" , "General"
   Return the response in the STRICT JSON format,
   {
   "audience_options" : array of strings,
   "image_classification" : string,
   } 
   """
   response = aud_mod.generate_content([file_path, prompt])
   return parse_gemini_response(response.text)

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  
            json_output = json_output.split("```")[0]  
            break  
    return json_output

def get_bbox(prompt, img):
    response = obj_det_model_2d.generate_content([prompt, img])
    return parse_json(response.text)

def plot_bounding_boxes(im, bounding_boxes):

    img = im.copy()
    width, height = img.size

    draw = ImageDraw.Draw(img)

    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
        'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime',
        'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet',
        'gold', 'silver',
    ] + additional_colors

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
      color = colors[i % len(colors)]

      abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
      abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
      abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
      abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 12), bounding_box["label"], fill="black", font=font)
    
    return img

def get_points(prompt, img):
    response = obj_det_model_1d.generate_content([prompt, img])
    return parse_json(response.text)

def plot_points(im, points_json):
    img = im.copy()  
    width, height = img.size
    draw = ImageDraw.Draw(img)

    colors = [
        'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
        'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta', 'lime',
        'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender', 'violet',
        'gold', 'silver',
    ] + additional_colors

    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    points_data = json.loads(points_json)

    for i, point_data in enumerate(points_data):
        color = colors[i % len(colors)]  

        abs_y, abs_x = int(point_data["point"][0] / 1000 * height), int(point_data["point"][1] / 1000 * width)

        radius = 10
        draw.ellipse((abs_x - radius, abs_y - radius, abs_x + radius, abs_y + radius), fill=color, outline=color)

        if "label" in point_data:
            draw.text((abs_x + 8, abs_y + 12), point_data["label"], fill="black", font=font)

    return img

st.set_page_config(page_title="Image Insights AI", layout="wide")

if 'current_img' not in st.session_state:
    st.session_state.current_img = {'img_id' : None, 'path' : None,'image' : None, 'type' : None}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'audience' not in st.session_state:
    st.session_state.audience = "General"
if 'select_aud' not in st.session_state:
    st.session_state.select_aud = None

st.title('Image Insights AI')

with st.sidebar:
    st.title("Image Insights AI")

    option = st.selectbox("Choose way of uploading an image", 
                        ["Upload the file", "Paste the link of the Image"])

    if option == "Upload the file":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
        if uploaded_file:
            try:
                st.session_state.current_img['path'] = get_from_uploads(uploaded_file)
                st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
            except Exception as e:
                st.error(f"Failed to upload image: {str(e)}")
        
        img_id = f"{uploaded_file}_id"

    else:
        img_url = st.text_input("Paste the link of the image file", help= "Paste the image link")
        if img_url:
            try:
                st.session_state.current_img['path'] = load_img(img_url)
                st.session_state.current_img['image'] = Image.open(st.session_state.current_img['path'])
                st.image(st.session_state.current_img['image'], caption='Uploaded Image', use_container_width=True)

            except Exception as e:
                st.error(f"Failed to download image: {str(e)}")
        
        img_id = f"{img_url}_id"
    
    if img_id and st.session_state.current_img['path']:
        with st.expander("Image ID extraction", expanded = True):
            if (st.session_state.current_img['img_id'] != img_id):
                st.session_state.chat_history[img_id] = []
                st.session_state.current_img['img_id'] = img_id
                st.session_state.current_img['image'] = Image.open(st.session_state.current_img['path'])
    
        if st.button("Process Image"):
            try:
                with st.spinner("Getting Details"):
                    resp_json = get_about(st.session_state.current_img['image'])
                    st.session_state.current_img['type'] = resp_json['image_classification']
                    st.session_state.audience = resp_json['audience_options'] if resp_json['audience_options'] else ["General"]

                    st.success("Image Processed Successfully")
            
            except Exception as e:
                st.error(f"Failed to process image: {str(e)}")

if st.session_state.current_img['type'] and st.session_state.audience:
    st.session_state.selected_aud = st.selectbox(
        "Select Audience" ,
        options = st.session_state.audience,
        key = 'audience_selector'
    )

    if st.session_state.current_img['type'] == "Educational":
        menu_options = ["Summary", "Roadmap", "Object Detection", "Chat"]
        st.info("This Image is related to Educational Content")

    else:
        menu_options = ["Summary", "Similar Content", "Object Detection", "Chat"]
        st.info("This Image is related to Entertainment Content")

    selected = option_menu(
        menu_title=None,
        options=menu_options,
        icons=["file-text", "map" if st.session_state.current_img['type'] == "Educational" else "film", 
            "card-text", "chat"],
        default_index=0,
        orientation="horizontal"
    )

    def summary_prompt(aud):
        summary_prompt = f"Provide the detailed information and summary about the content in the provided image with respect to the set of audience : {aud}"
        return summary_prompt

    def roadmap_prompt(aud):
        roadmap_prompt = f"For the provided image related to educational content, provide a roadmap to learn about the topic related to given image, with respect to the set of audience : {aud}"
        return roadmap_prompt

    def sim_content_prompt(aud):
        sim_content_prompt = f"For the provided image, related to General audience, provide the information related to the similar content (means, similar images, related topis, etc..) related to the image, with respect to the set of audience : {aud}"
        return sim_content_prompt

    if selected == "Summary":
        st.subheader("Summary Generator")
        if st.button("Generate Summary"):
            with st.spinner("Generating the Summary...") :
                response = img_model.generate_content([summary_prompt(st.session_state.select_aud), st.session_state.current_img['image']])
                st.write(response.text)

    elif selected == "Roadmap" and st.session_state.current_img['type'] == "Educational":
        st.subheader("Roadmap Generator")
        if st.button("Generate Roadmap"):
            with st.spinner("Generating the Roadmap...") :
                response = img_model.generate_content([roadmap_prompt(st.session_state.select_aud), st.session_state.current_img['image']])
                st.write(response.text)

    elif selected == "Similar Content" and st.session_state.current_img['type'] != "Educational":
        st.subheader("Similar Content Generator")
        if st.button("Get Similar Content"):
            with st.spinner("Generating Similar Content...") :
                response = img_model.generate_content([sim_content_prompt(st.session_state.select_aud), st.session_state.current_img['image']])
                st.write(response.text)
    
    elif selected == "Chat":
        st.subheader("Image Chat Assistant")
        current_chat = st.session_state.chat_history.get(img_id, [])
        
        for message in current_chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about the content"):
            current_chat.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = img_model.generate_content(["Answer the Question from the given image with respect to the given set of Audience, while answering certain questions you can also use the knowledge from outside the image too (but just verify whether it is related to the image)\n"
                    f"Audience: {st.session_state.select_aud}\n"
                    f"Question: {prompt}\n",
                    st.session_state.current_img['image']
                ])

                current_chat.append({"role": "assistant", "content": response.text})
            
            st.session_state.chat_history[img_id] = current_chat
            
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)

    elif selected == "Object Detection":
        st.subheader("Object Detection")
        with st.expander("Dimension Choosing", expanded = True):
            choice = st.radio("Choose the Dimension of the bounding box", options = ["Point (1D)", "Square (2D)"])

        if choice == "Square (2D)":
            user_prompt = st.text_input(label = "Enter your preference of object detection", value = "Plot for all the objects/entities present in the image", help="Make your object detection dynamic")
            if st.button("Detect Objects"):
                with st.spinner("Detecting Objects"):
                    bboxes = get_bbox(user_prompt, st.session_state.current_img['image'])
                    st.image(plot_bounding_boxes(st.session_state.current_img['image'], bboxes), caption = "Detected Objects", width = 800)

        elif choice == "Point (1D)":
            user_prompt = st.text_input(label = "Enter your preference of object detection", value = "Plot for all the objects/entities present in the image", help="Make your object detection dynamic")
            if st.button("Detect Objects"):
                with st.spinner("Detecting Objects"):
                    bboxes = get_points(user_prompt, st.session_state.current_img['image'])
                    st.image(plot_points(st.session_state.current_img['image'], bboxes), caption = "Detected Objects", width = 800)

else:
    st.warning("Please upload an image")