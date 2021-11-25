/*
   XmlDomDocument.cpp

   DOM parsing class definitions.

   ------------------------------------------

   Copyright (c) 2013 Vic Hargrave

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include "xmldom/XmlDomDocument.h"

class XmlDomErrorHandler : public HandlerBase
{
  public:
    void fatalError(const SAXParseException &exc) {
        printf("Fatal parsing error at line %d\n", (int)exc.getLineNumber());
        exit(-1);
    }
};

XercesDOMParser*   parser = NULL;
ErrorHandler*      errorHandler = NULL;

void createParser()
{
    if (!parser)
    {
        XMLPlatformUtils::Initialize();
        parser = new XercesDOMParser();
        errorHandler = (ErrorHandler*) new XmlDomErrorHandler();
        parser->setErrorHandler(errorHandler);
    }
}

XmlDomDocument::XmlDomDocument(const char* xmlfile) : m_doc(NULL)
{
    createParser();
    parser->parse(xmlfile);
    m_doc = parser->adoptDocument();
}

XmlDomDocument::~XmlDomDocument()
{
    if (m_doc) m_doc->release();
}

string XmlDomDocument::getChildValue(const char* parentTag, int parentIndex, const char* childTag, int childIndex)
{
	XMLCh* temp = XMLString::transcode(parentTag);
	DOMNodeList* list = m_doc->getElementsByTagName(temp);
	XMLString::release(&temp);

	DOMElement* parent = dynamic_cast<DOMElement*>(list->item(parentIndex));
	DOMElement* child = 
        dynamic_cast<DOMElement*>(parent->getElementsByTagName(XMLString::transcode(childTag))->item(childIndex));
	string value;
	if (child) {
		char* temp2 = XMLString::transcode(child->getTextContent());
		value = temp2;
        XMLString::release(&temp2);
	}
	else {
		value = "";
	}
	return value;
}

string XmlDomDocument::getChildAttribute(const char* parentTag, int parentIndex, const char* childTag, int childIndex,
                                         const char* attributeTag)
{
	XMLCh* temp = XMLString::transcode(parentTag);
	DOMNodeList* list = m_doc->getElementsByTagName(temp);
	XMLString::release(&temp);

	DOMElement* parent = dynamic_cast<DOMElement*>(list->item(parentIndex));
	DOMElement* child = 
        dynamic_cast<DOMElement*>(parent->getElementsByTagName(XMLString::transcode(childTag))->item(childIndex));
	string value;
	if (child) {
        temp = XMLString::transcode(attributeTag);
		char* temp2 = XMLString::transcode(child->getAttribute(temp));
		value = temp2;
        XMLString::release(&temp2);
	}
	else {
		value = "";
	}
	return value;
}

int XmlDomDocument::getRootElementCount(const char* rootElementTag)
{
	DOMNodeList* list = m_doc->getElementsByTagName(XMLString::transcode(rootElementTag));
	return (int)list->getLength();
}

int XmlDomDocument::getChildCount(const char* parentTag, int parentIndex, const char* childTag)
{
	XMLCh* temp = XMLString::transcode(parentTag);
	DOMNodeList* list = m_doc->getElementsByTagName(temp);
	XMLString::release(&temp);

	DOMElement* parent = dynamic_cast<DOMElement*>(list->item(parentIndex));
	DOMNodeList* childList = parent->getElementsByTagName(XMLString::transcode(childTag));
    return (int)childList->getLength(); 
}
