/*
   XmlDomDocument.cpp

   DOM parsing class interfaces.

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

#ifndef __XmlDomDocument_h__
#define __XmlDomDocument_h__

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

#include <string>

using namespace std;
using namespace xercesc;

class XmlDomDocument
{
    DOMDocument* m_doc;

  public:
    XmlDomDocument(const char* xmlfile);
    ~XmlDomDocument();

    string getChildValue(const char* parentTag, int parentIndex, const char* childTag, int childIndex);
    string getChildAttribute(const char* parentTag, int parentIndex, const char* childTag, int childIndex,
                             const char* attributeTag);
    int getRootElementCount(const char* rootElementTag);
    int getChildCount(const char* parentTag, int parentIndex, const char* childTag);

  private:
    XmlDomDocument();
    XmlDomDocument(const XmlDomDocument&);
};

#endif
