import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_bytes,convert_from_path
import pymupdf # imports the pymupdf library

import pandas as pd
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from fpdf import FPDF
from PIL import Image as PILImage
import os
from openai import AzureOpenAI

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
# import fitz
import tempfile 
from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
import io
import streamlit as st
# Set up environment for Tesseract and LangChain

os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"

# Azure Blob Storage setup
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=translatepdf;AccountKey=ElInSAO52198hJaZW9bp5olaKtAdme66HyGb82j/0PgFX2ufhpnRN3iVZ/fbaqqjhZuBj8xChOB6+AStG4vrzA==;EndpointSuffix=core.windows.net")
russian_container = "russiandocs"
english_container = "englishtranslateddoc"
api_key = st.secrets["general"]["AZURE_OPENAI_API_KEY"]
endpoint = st.secrets["general"]["AZURE_OPENAI_ENDPOINT"]
deployment_name = st.secrets["general"]["AZURE_OPENAI_DEPLOYMENT_NAME"]
# Initialize LangChain LLM
def llm(text,language) -> str:
    client = AzureOpenAI(
            api_key =api_key ,  
            api_version = "2025-01-01-preview",
            azure_endpoint = endpoint
            )
    prompt = (
        f"You will be provided with a text in {language}: \"{text}\". "
        
        "Translate it into English while maintaining spaces, new lines, and structure."
    )

    response = client.chat.completions.create(
        model=deployment_name,  # your Azure deployment name here
        messages=[
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content.strip()

# app = FastAPI()

# Azure Blob Storage helper fu nctions
def upload_pdf_to_blob(pdf_data: BytesIO, container_name: str, blob_name: str):
    """
    Upload the in-memory PDF to the specified Azure Blob container, if it doesn't already exist.
    
    :param pdf_data: The in-memory PDF data as a BytesIO object.
    :param container_name: The name of the Azure Blob container.
    :param blob_name: The name of the blob to be created or overwritten.
    :return: The URL of the uploaded blob.
    """
    # Get the container client and blob client
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)

    # Check if the blob already exists
    if blob_client.exists():
        print(f"The file '{blob_name}' already exists in the container '{container_name}'. Skipping upload.")
        return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
    
    # If the blob doesn't exist, upload the in-memory PDF
    if hasattr(pdf_data, 'seek'):
        pdf_data.seek(0)  # Rewind the BytesIO object to the beginning
    
    blob_client.upload_blob(pdf_data, overwrite=True)

    print(f"File '{blob_name}' uploaded to container '{container_name}' successfully.")
    
    return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"

# Processing and translating text from OCR
def process_and_translate(ocr_text, ocr_data,language):
    empty_count = 0
    previous_text = ""
    table_placeholder_added = False  # To track if {table} placeholder has been added
    
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i]
        
        if text == "" or text == " ":
            empty_count += 1
        else:
            if empty_count >= 5 and not table_placeholder_added:  # Add {table} placeholder before the first detected text
                ocr_text = ocr_text.replace(previous_text, f"{previous_text} {{image}}", 1)
                table_placeholder_added = True
            previous_text = text
            empty_count = 0

    if empty_count >= 5 and not table_placeholder_added:
        ocr_text = ocr_text.replace(previous_text, f"{previous_text} {{image}}", 1)

    answer = llm(ocr_text,language)
    return answer

# Translate batch of words
def process_and_translate_batch(words):
    prompt_template_batch = """You will be provided with a list of words or phrases in Russian language {Russian_words}. 
                               Use your knowledge and translate these into English. 
                               ONLY PROVIDE THE TRANSLATED TEXT, NO SUGGESTIONS, NO SENTENCES. 
                               Provide the translations in a list format as per the order of the input."""
    
    words_str = ", ".join(words)
    prompt_batch = PromptTemplate(template=prompt_template_batch, input_variables=["Russian_words"])
    chain_batch = LLMChain(llm=llm, prompt=prompt_batch)
    
    translations = chain_batch.run(words_str)
    translated_words = translations.split('\n')  # Split by newline if translations are returned line-by-line
    translated_words = [word.strip() for word in translated_words if word.strip()]
    
    return translated_words

# Extract table from page
def extract_table(image_path,selected_lang_code):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    dilated_image = cv2.dilate(threshold_image, kernel, iterations=1)

    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    horizontal_lines = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel_v)

    table_structure = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cv2.boundingRect(contour) for contour in contours]
    cells = sorted(cells, key=lambda x: (x[1], x[0]))

    filtered_cells, table_main = filter_detected_cells(cells)

    if table_main:
        x_t, y_t, w_t, h_t = table_main
        image_with_table = image.copy()
        cv2.rectangle(image_with_table, (x_t, y_t), (x_t + w_t, y_t + h_t), (0, 255, 0), 2)
        cv2.imwrite("image_with_table_outline.png", image_with_table)

    ocr_data = pytesseract.image_to_data(image, lang=selected_lang_code, output_type=pytesseract.Output.DICT)
    table_text = []
    outside_text = []
    df = pd.DataFrame({'text': ocr_data['text'], 'left': ocr_data['left'], 'top': ocr_data['top'], 'width': ocr_data['width'], 'height': ocr_data['height'], 'confidence': ocr_data['conf']})

    for i, row in df.iterrows():
        text = row["text"].strip()
        if text:
            x = row["left"]
            y = row["top"]
            w = row["width"]
            h = row["height"]
            if x_t <= x and x + w <= x_t + w_t and y_t <= y and y + h <= y_t + h_t:
                table_text.append((x, y, w, h, text))
            else:
                outside_text.append((x, y, w, h, text))

    cell_text_mapping = {cell: [] for cell in filtered_cells}

    for x_text, y_text, w_text, h_text, text in table_text:
        for x_cell, y_cell, w_cell, h_cell in filtered_cells:
            if x_cell <= x_text <= x_cell + w_cell and y_cell <= y_text <= y_cell + h_cell:
                cell_text_mapping[(x_cell, y_cell, w_cell, h_cell)].append(text)
                break

    structured_table = [(x, y, w, h, " ".join(texts)) for (x, y, w, h), texts in cell_text_mapping.items()]
    rows = {}
    for x, y, w, h, text in structured_table:
        row_key = min(rows.keys(), key=lambda k: abs(k - y)) if rows else y
        if abs(row_key - y) > 10:
            row_key = y
        rows.setdefault(row_key, []).append((x, text))

    table_data = [sorted(row, key=lambda x: x[0]) for row in rows.values()]
    df = pd.DataFrame([[cell[1] for cell in row] for row in table_data])

    # Translate the extracted table using process_and_translate_table
    translated_df = df.copy()

    words_to_translate = []
    for col in df.columns:
        for row in range(len(df)):
            text = df.at[row, col]
            if isinstance(text, str) and text.strip():
                words_to_translate.append(text)

    translated_words = process_and_translate_batch(words_to_translate)

    translated_df = df.copy()

    translated_index = 0
    for col in df.columns:
        for row in range(len(df)):
            text = df.at[row, col]
            if isinstance(text, str) and text.strip():
                translated_df.at[row, col] = translated_words[translated_index]
                translated_index += 1

    output_path = "translated_table_new.png"
    save_table_as_image(translated_df, output_path)
    return output_path

# Save DataFrame as image
def save_table_as_image(df, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=[f"Col {i}" for i in range(df.shape[1])], cellLoc='center', loc='center')
    plt.savefig(output_path)
    plt.close()
def image_extraction(ocr_data,image,image_width):
    # Initialize variables for empty text sequence detection
    empty_count = 0
    start_idx = None
    empty_regions = []  # List to store start and end indices of empty text regions

    # Iterate through the OCR data to find consecutive empty text regions
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()

        if text == "" or text == " ":  # If empty text (possible diagram region)
            if empty_count == 0:
                # Record the starting index of the empty text sequence
                start_idx = i
            empty_count += 1
        else:
            if empty_count >= 5:  # If we have more than 15 consecutive empty texts
                # Record the end index of the empty text sequence
                end_idx = i

                # Add the start and end indices of this empty region to the list
                empty_regions.append((start_idx, end_idx))

            # Reset empty count for next sequence
            empty_count = 0

    # If the sequence ends with empty text, we need to check it after the loop ends
    if empty_count >= 5:
        empty_regions.append((start_idx, len(ocr_data['text'])))

    # Now we have the empty regions with start and end indices where empty text spans
    # Loop through the empty regions to create a bounding box that covers everything between `start_idx` and `end_idx`
    for start_idx, end_idx in empty_regions:
        # Initialize the min_y and max_y to create a bounding box
        min_y = float('inf')  # To store the top coordinate
        max_y = 0  # To store the bottom coordinate

        # Loop through the OCR data for indices from start_idx to end_idx
        for idx in range(start_idx, end_idx):
            y, h = ocr_data['top'][idx], ocr_data['height'][idx]    

            # Update the min_y and max_y values to encompass everything between start_idx and end_idx
            min_y = min(min_y, y)
            max_y = max(max_y, y + h)

        # Now you have the min_y and max_y coordinates, and you can extract the image
        x_start = 0  # Starting at the left edge of the image
        x_end = image_width  # Ending at the right edge of the image
        # min_y=min_y-35
        max_y=max_y-60
        image_np = np.array(image)

        # Extract the region of interest (ROI) from the image
        roi = image_np[min_y:max_y, x_start:x_end]

        pil_image = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Save the image to a BytesIO object (in memory)
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)  # Rewind the BytesIO object to the beginning

        return img_byte_arr
# Create translated page with text and table
def create_translated_page(page, translated_text, table_image_path):
    translated_page = translated_text
    if table_image_path:
        translated_page += f"\n[Table Image: {table_image_path}]"
    return translated_page

# Filter detected table cells to remove small cells
def filter_detected_cells(cells):
    largest_area = 0
    table_outline = None

    for cell in cells:
        x, y, w, h = cell
        area = w * h
        if area > largest_area:
            largest_area = area
            table_outline = cell

    if table_outline:
        cells.remove(table_outline)

    filtered_cells = [(x, y, w, h) for x, y, w, h in cells if w >= 5 and h >= 5]

    return filtered_cells, table_outline

# Combine all translated pages into a final PDF
def combine_pages_into_pdf(translated_pages, image_streams_list):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    for page_num, page_content in enumerate(translated_pages):
        pdf.add_page()
        if isinstance(page_content, list):
            page_content = "\n".join(page_content)
        pdf.multi_cell(0, 10, page_content)

        if page_num < len(image_streams_list):
            image_stream = image_streams_list[page_num]
            if image_stream is not None:
                try:
                    pil_image = PILImage.open(image_stream)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
                        pil_image.save(tmp_img_file.name)
                        tmp_img_path = tmp_img_file.name

                    pdf.image(tmp_img_path, x=10, y=pdf.get_y(), w=100)
                    pdf.ln(50)

                    os.remove(tmp_img_path)
                except Exception as e:
                    print(f"Skipping image for page {page_num} due to error: {e}")

    # Write PDF to a temporary file, then load into BytesIO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        pdf.output(temp_pdf_file.name)
        temp_pdf_file.seek(0)
        pdf_output = BytesIO(temp_pdf_file.read())

    # Clean up temp PDF file
    try:
        os.remove(temp_pdf_file.name)
    except Exception as e:
        print(f"Warning: Could not delete temp PDF file: {e}")

    pdf_output.seek(0)
    return pdf_output
    # return pdf  # Return the PDF as a BytesIO stream
def replace_image_in_text(text, image_streams):
    """
    Replace {Image} placeholder with actual images from the list of image streams (in-memory).
    The images are added directly in place of the {Image} placeholder.
    
    :param text: The text with {Image} placeholders
    :param image_streams: List of in-memory image data (BytesIO objects)
    :return: Updated text and image streams
    """
    # image_np = np.array(image_streams)

    text_parts = text.split("{Image}")
    final_text = []
    image_streams_list = []  # List to store image streams
    
    # Combine the text and images (in-memory)
    for i, part in enumerate(text_parts):
        final_text.append(part)  # Add the text part
        
        
         # In-memory image stream (BytesIO)
        image_streams_list.append(image_streams)  # Store the image stream
    
    return final_text, image_streams_list
# Process PDF and generate translated PDF
def process_pdf_and_translate(pdf_data,selected_lang_code,zoomlevel=4):
    # pdf_document = fitz.open(pdf_path)
    translated_pages = []
    lstimages = []

    zoom_x = zoomlevel  # horizontal zoom
    zoom_y = zoomlevel  # vertical zoom
    mat = pymupdf.Matrix(zoom_x, zoom_y)
    doc = pymupdf.open(stream=pdf_data) 
    for page_num in range(doc.page_count):  # iterate through all the pages
            page = doc.load_page(page_num)  # load the current page
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            
            # Convert pixmap to a PIL Image object
            img_bytes = pix.tobytes("png")  # Convert to PNG bytes
            img = Image.open(io.BytesIO(img_bytes))  # Open the bytes as a PIL Image
            
            # Append the image object to the list for further processing
            lstimages.append(img)
    # pages = convert_from_path(pdf_path,poppler_path="C:/Users/mihir.sinha/Downloads/Release-24.08.0-0/poppler-24.08.0/Library/bin")
    # Loop through each page
    for idx, img in enumerate(lstimages):
        

        image = img
        image_width, image_height = img.size  # Get width and height of the image

        # Perform OCR on the image
        ocr_text = pytesseract.image_to_string(image, lang=selected_lang_code)
        ocr_data= pytesseract.image_to_data(image, lang=selected_lang_code,output_type=pytesseract.Output.DICT)
        if selected_lang_code == 'eng':
            language = 'English'
        elif selected_lang_code == 'fra':
            language = 'French'
        elif selected_lang_code == 'jpn':
            language = 'Japanese'
        elif selected_lang_code == 'rus':
            language = 'Russian'
        elif selected_lang_code == 'spa':
            language = 'Spanish'
        else:
            language = 'Unknown'
        # Process and translate text (both normal text and tables)
        translated_text = process_and_translate(ocr_text,ocr_data,language)
        # Step 2: Extract images from empty regions
        extracted_image_lst = []
        extracted_image = image_extraction(ocr_data,image,image_width)  # Extract image regions
        if extracted_image:
            extracted_image_lst.append(extracted_image)

#     # Step 3: Replace {Image} placeholders with actual images
        text_with_images,img_stream_lst = replace_image_in_text(translated_text, extracted_image)

#     # Step 4: Create the final PDF with text and images
#     pdf_output = create_pdf_with_images(text_with_images)
        # translated_page = create_translated_page(page, translated_text, table_image_path)
        translated_pages.append(text_with_images)
    
    # After processing all pages, combine them into a final PDF
    final_pdf = combine_pages_into_pdf(translated_pages,img_stream_lst)

    return final_pdf

# # FastAPI Endpoint
# @app.post("/process-pdf/")
def process_pdf(file: UploadFile ,selected_lang_code):
    # Save uploaded file temporarily
    # with tempfile.NamedTemporaryFile(delete=True,mode='wb') as tmp_file:
    #     tmp_file.write(await file.read())
    #     tmp_pdf_path = tmp_file.name
    pdf_data = file.read()
    pdf_stream = BytesIO(pdf_data)
    # Upload the PDF to the 'russiandoc' container
    upload_pdf_to_blob(pdf_stream, russian_container, file.name)
    container_client = blob_service_client.get_container_client(russian_container)
    blob_client = container_client.get_blob_client(file.name)
    
     
    # stream = io.BytesIO()

    blob_data = blob_client.download_blob() # This returns the actual bytes

# Use io.BytesIO to treat the bytes as a file-like object
    pdf_data = blob_data.readall() #     outfilename = "output.pdf"
    # outPDFpath = os.path.join(tempfile.gettempdir(), outfilename)

# # Step 2: Write the PDF data to a temporary file
#     with open(outPDFpath, "wb") as tmp_file:
#     # Step 3: Copy the content from pdf_stream to the temporary file
#         tmp_file.write(pdf_stream.read()) 
 # Read the blob data and load it into a BytesIO object
    # temp_dir="D:/OneDrive - Beyond Key Systems Pvt. Ltd/Transaltion"
    # with tempfile.NamedTemporaryFile(delete=True, mode='wb') as tmp_file:
    #     tmp_file.write(pdf_data)  # Write the downloaded PDF content to the temp file
    #     new_tmp_pdf_path = tmp_file.name
    # pdf=pdf_data.__bytes__
    
    # return pdf_data
    # Process the PDF and get the translated PDF path
    
    final_pdf_path = process_pdf_and_translate(pdf_data,selected_lang_code)

    # Upload the translated PDF to the 'englishtranslateddoc' container
    upload_pdf_to_blob(final_pdf_path, english_container, f"translated_{file.name}")

    # Return the link to the translated PDF
    return {"translated_pdf_url": f"https://{blob_service_client.account_name}.blob.core.windows.net/{english_container}/translated_{file.name}"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app)
