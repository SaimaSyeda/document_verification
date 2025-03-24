# In models/text_extractor.py
import pytesseract
import cv2
import re

class TextExtractor:
    def __init__(self):
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass
        
    def extract_text(self, image):
        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)
        return text
    
    def extract_structured_data(self, text, document_type):
        if document_type == "land_record":
            return self._extract_land_record_data(text)
        elif document_type == "caste_certificate":
            return self._extract_caste_certificate_data(text)
        elif document_type == "property_registration":
            return self._extract_property_registration_data(text)
        else:
            return {"error": "Unsupported document type"}
    
    def _extract_land_record_data(self, text):
        data = {}
        
        # Extract property ID
        property_id_match = re.search(r'Property ID[:\s]+([A-Z0-9-]+)', text, re.IGNORECASE)
        if property_id_match:
            data['property_id'] = property_id_match.group(1)
        
        # Extract owner name
        owner_match = re.search(r'Owner Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if owner_match:
            data['owner_name'] = owner_match.group(1).strip()
        
        # Extract area
        area_match = re.search(r'Area[:\s]+([\d.]+)\s*([A-Za-z²]+)', text, re.IGNORECASE)
        if area_match:
            data['area'] = area_match.group(1)
            data['area_unit'] = area_match.group(2)
        
        # Extract location
        location_match = re.search(r'Location[:\s]+([A-Za-z0-9\s,.-]+)', text, re.IGNORECASE)
        if location_match:
            data['location'] = location_match.group(1).strip()
            
        return data
    
    # Similar methods for other document types...
    def _extract_caste_certificate_data(self, text):
        data = {}
        
        # Extract certificate number
        cert_match = re.search(r'Certificate No[:\s]+([A-Z0-9-/]+)', text, re.IGNORECASE)
        if cert_match:
            data['certificate_number'] = cert_match.group(1)
        
        # Extract person name
        name_match = re.search(r'Name[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if name_match:
            data['name'] = name_match.group(1).strip()
        
        # Extract caste
        caste_match = re.search(r'Caste[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if caste_match:
            data['caste'] = caste_match.group(1).strip()
            
        # Extract issuing authority
        authority_match = re.search(r'Issuing Authority[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if authority_match:
            data['issuing_authority'] = authority_match.group(1).strip()
            
        return data
        
    def _extract_property_registration_data(self, text):
        data = {}
        
        # Extract registration number
        reg_match = re.search(r'Registration No[:\s]+([A-Z0-9-/]+)', text, re.IGNORECASE)
        if reg_match:
            data['registration_number'] = reg_match.group(1)
        
        # Extract buyer name
        buyer_match = re.search(r'Buyer[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if buyer_match:
            data['buyer'] = buyer_match.group(1).strip()
        
        # Extract seller name
        seller_match = re.search(r'Seller[:\s]+([A-Za-z\s]+)', text, re.IGNORECASE)
        if seller_match:
            data['seller'] = seller_match.group(1).strip()
        
        # Extract property details
        property_match = re.search(r'Property Details[:\s]+([A-Za-z0-9\s,.-]+)', text, re.IGNORECASE)
        if property_match:
            data['property_details'] = property_match.group(1).strip()
            
        return data