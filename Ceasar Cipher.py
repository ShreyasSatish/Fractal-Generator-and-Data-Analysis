def ceasar_decode(string, offset = 10):
    # Decode a string using a Caesar cipher with the given offset.
    # If offset is 0 or 26, return the original string.
    if offset == 0 or offset % 26 == 0:
        return string
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encoded_message = ""
    for char in string: # Loop through each character in the string to decode it
        index = alphabet.find(char)
        if index != -1:
            new_index = (index - offset) % len(alphabet) # Calculate the new index by subtracting the offset and wrapping around
            encoded_message += alphabet[new_index]
        else: # Skip any characters not found in the alphabet, let the user know
            # print(f"Character '{char}' not found in alphabet, skipping.")
            encoded_message += char
    return encoded_message

def ceasar_encode(string, offset = 10):
    # Encode a string using a Caesar cipher with the given offset.
    # If offset is 0 or 26, return the original string.
    if offset == 0 or offset == 26:
        return string
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encoded_message = ""
    for char in string: # Loop through each character in the string to encode it
        index = alphabet.find(char)
        if index != -1:
            new_index = (index + offset) % len(alphabet) # Calculate the new index by adding the offset and wrapping around
            encoded_message += alphabet[new_index]
        else: # Skip any characters not found in the alphabet, let the user know
            # print(f"Character '{char}' not found in alphabet, skipping.")
            encoded_message += char
    return encoded_message

def vigenere_decode(string, key):
    # Decode a string using a Vigenère cipher with the given key.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encoded_message = ""
    key_length = len(key)
    keyword_phrase = ""
    j = 0
    for i in range(len(string)):
        if string[i] in alphabet:
            key_index = j % key_length
            keyword_phrase += key[key_index]  # Create a repeating keyword phrase based on the key
            j += 1
        else:
            keyword_phrase += string[i]  # Append non-alphabet characters as is
    
    resulting_place_value = []
    for i in range(len(string)):
        if string[i] in alphabet:
            index = alphabet.find(string[i])
            offset = alphabet.find(keyword_phrase[i])
            new_index = (index - offset) % len(alphabet)  # Calculate the new index
            resulting_place_value.append(alphabet[new_index])
        else:
            resulting_place_value.append(string[i])

    for item in resulting_place_value:
        if type(item) != int:
            encoded_message += item
        else:
            encoded_message += alphabet[item]
    return encoded_message

def vigenere_encode(string, key):
    # Encode a string using a Vigenère cipher with the given key.
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    encoded_message = ""
    key_length = len(key)
    keyword_phrase = ""
    j = 0
    for i in range(len(string)):
        if string[i] in alphabet:
            key_index = j % key_length
            keyword_phrase += key[key_index]  # Create a repeating keyword phrase based on the key
            j += 1
        else:
            keyword_phrase += string[i]  # Append non-alphabet characters as is

    resulting_place_value = []
    for i in range(len(string)):
        if string[i] in alphabet:
            index = alphabet.find(string[i])
            offset = alphabet.find(keyword_phrase[i])
            new_index = (index + offset) % len(alphabet)  # Calculate the new index
            resulting_place_value.append(alphabet[new_index])
        else:
            resulting_place_value.append(string[i])

    for item in resulting_place_value:
        if type(item) != int:
            encoded_message += item
        else:
            encoded_message += alphabet[item]
    return encoded_message

