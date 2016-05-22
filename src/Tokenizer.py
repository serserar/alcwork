    # coding=utf-8
    #
    # <one line to give the program's name and a brief idea of what it does.>
    # Copyright (C) 2016  <copyright holder> <email>
    #
    # This program is free software; you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation; either version 2 of the License, or
    # (at your option) any later version.
    #
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License along
    # with this program; if not, write to the Free Software Foundation, Inc.,
    # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
    #
    #
    # Construir un tokenizador para el espanol, que, dado un fichero de texto de entrada
    # (entrada_tokenizador.txt),separe en tokens,y lo muestre en un fichero	de salida en el
    # formato que se muestra en (salida_tokenizador.txt). Se deben cumplir las siguientes
    # restricciones

import re

class Tokenizer:

        def __init__(self):
            self.emoticons = "(?:[:=;][oO\-]?[D\)\]\(\]/\\OpP])"
            self.mentions="(?:@[\w_]+)"
            self.whitespace = "\s+"
            self.symbols = "[#().'\";:]"
            self.wordsRegex = "\w+"
            self.numbersRegex = "[-+]?[0-9]*\.?[0-9]+"#"[-+]?\d*\.\d+|\d+"
            self.emailRegex = "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+" #"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
            self.urlRegex = "http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+"#"http[s]?://\S+"
            self.dateRegex = "[\d]+[/][\d]+[/][\d]+|[\d]+[-][\d]+[-][\d]+"
            self.timeRegex = "[\d]+[:][\d]+[h]*|[\d]+[:][\d]+\s[h]*|[\d]+[:][\d]+[\s][horas]"
            self.regex = "(%s)|(%s)|(%s)|(%s)|(%s)|(%s)|(%s)|(%s)|(%s)|(%s)" % (self.timeRegex,
            self.dateRegex, self.emailRegex, self.urlRegex, self.numbersRegex,
            self.whitespace, self.mentions, self.emoticons, self.symbols, self.wordsRegex)
            self.expression = re.compile(self.regex, re.DOTALL | re.VERBOSE | re.LOCALE ) # | re.UNICODE)
            pass
        
        def tokenize_file(self, textPath):
          f = open(textPath, 'r')
          lines = []
          for line in f:
            lines.append(self.tokenize(line, True))
          return lines
      
        def cleanText(self, text):    
            tokens = self.tokenize(text, True)
            clean_text = ""
            for token in tokens:
                clean_text += token
            return clean_text
        
        def tokenize(self, text, lowercase = False):
            tokens = []
            for match in re.finditer(self.expression, text):
              self.timeRegex, self.dateRegex, self.emailRegex, self.urlRegex, self.numbersRegex , self.whitespace, self.mentions, self.emoticons, self.symbols, self.wordsRegex = match.groups()
              token = match.group(0)
              if not self.emoticons:
                  token = token.lower()
              elif self.whitespace:
                  token = " "    
              tokens.append(token)
            return tokens  
              
        def tokenize_debug(self, text):
            tokens = []
            for match in re.finditer(self.expression, text):
              self.timeRegex, self.dateRegex, self.emailRegex, self.urlRegex, self.numbersRegex, self.whitespace, self.mentions, self.emoticons, self.symbols, self.wordsRegex = match.groups()
              tokens.append(match.group(0))
              if self.emoticons: print("emoticons : " + match.group(0))
              if self.mentions: print("mentions : " + match.group(0))
              if self.whitespace: print("whitespace : " + match.group(0))
              if self.symbols: print ("symbols : " + match.group(0))
              if self.wordsRegex: print("wordsRegex : " + match.group(0))
              if self.numbersRegex: print("numbersRegex : " + match.group(0))
              if self.emailRegex: print("emailRegex : " + match.group(0))
              if self.urlRegex: print("urlRegex : " + match.group(0))
              if self.dateRegex: print("dateRegex : " + match.group(0))
              if self.timeRegex: print("timeRegex : " + match.group(0))      
            return tokens
def main():
        tok = Tokenizer()
        print(tok.cleanText("@serserar :D, camión  test SOY YO sds 12/12/90 25-05-1979 1999 serserar@gmail.com, 199.123 http://www.google.es/serserar 9:30h"))
        #print(tok.tokenize_file("../entrada_tokenizador_UTF8.txt"))
if __name__ == '__main__':
    main()    