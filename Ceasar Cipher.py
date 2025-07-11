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

def main():
    # string = "xuo jxuhu! jxyi yi qd unqcfbu ev q squiqh syfxuh. muhu oek qrbu je tusetu yj? y xefu ie! iudt cu q cuiiqwu rqsa myjx jxu iqcu evviuj!"
    # offset = 10
    # encoded_message = decode(string, offset)
    # print(f"The encoded message is: {encoded_message}")

    # string = "managed to decode the message you sent! send me another one bitch!"
    # encoded_message = encode(string, offset)
    # print(f"The encoded message is: {encoded_message}")

    # print(decode(encoded_message, offset))

    # string = "jxu evviuj veh jxu iusedt cuiiqwu yi vekhjuud."
    # encoded_message = decode(string, 16)
    # print(f"The first encoded message is: {encoded_message}")

    # string = "bqdradyuzs ygxfubxq omqemd oubtqde fa oapq kagd yqeemsqe ue qhqz yadq eqogdq!"
    # encoded_message = decode(string, 12)
    # print(f"The second encoded message is: {encoded_message}")

    # string = "vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx."
    # for i in range(27):
    #     print(i)
    #     print(decode(string, i))

    # string = "txm srom vkda gl lzlgzr qpdb? fepb ejac! ubr imn tapludwy mhfbz cza ruxzal wg zztylktoikqq!"
    # key = "friends"
    # encoded_message = vigenere_decode(string, key)
    # print(encoded_message)
    
    string = "good job decoding the message, you’re a smart hamster. love you. now, if you completed this before the fifteenth of july i owe you whatever you want. if you say you want nothing then you’re getting a lunch and dessert on me. if you completed it after the aforementioned date, then you owe me a coffee. sounds fair right? good job completing this though, love you loads hyrax."
    offset = 7
    first_message = vigenere_encode(string, "devisha")
    print(f"{first_message}")
    print()
    second_message = "you should get to this point after doing the ceasar cipher correctly first. i thought i’d leave this as a checkpoint for you to make it a little easier, but your code should work if you’ve gotten here. keep going, almost there, only use the text after the colon for the next step otherwise it won't work: jsjl bvb gixwvpnj xcm elsvebm, qvu’ui v aehrw lvukaeu. pjdw fox. rje, am yry xwewlhxzl loiv fzngye wlz namthiibz vf mygg a vwh cjc ooawiqmj fox avvl. pf bsp asf yry rifa nrxcqfn tkii ggb’rh kzblpnj e gcfjh dry lwzshvo wf te. lj twm joptgmlld lx vnllr wlz ixvrhqzvlpoqiy lsae, wlzv qvu raz uw h crjamw. zoxrya xhiu vdoza? grsy rgi crqktwaiqk opaz tkspoz, soyi twm sodhn pqyaa."
    encoded_second_message = ceasar_encode(second_message, offset)
    print(f"{encoded_second_message}")
    print()
    print()
    print()

    string = "fvb zovbsk nla av aopz wvpua hmaly kvpun aol jlhzhy jpwoly jvyyljasf mpyza. p aovbnoa p’k slhcl aopz hz h joljrwvpua mvy fvb av thrl pa h spaasl lhzply, iba fvby jvkl zovbsk dvyr pm fvb’cl nvaalu olyl. rllw nvpun, hstvza aolyl, vusf bzl aol alea hmaly aol jvsvu mvy aol ulea zalw vaolydpzl pa dvu'a dvyr: qzqs ici npedcwuq ejt lszclit, xcb’bp c hloyd scbrhlb. wqkd mve. yql, ht fyf edldsoegs svpc mgunfl dsg uhtaoppig cm tfnn h cdo jqj vvhdpxtq mve hccs. wm izw hzm fyf ypmh uyejxmu arpp nni’yo rgiswuq l njmqo kyf sdgzocv dm al. sq adt qvwantssk se cussy dsg pecyoxgcswvxpf szhl, dsgc xcb yhg bd o jyqhtd. gveyfh eopb ckvgh? nyzf ynp jyxradhpxr vwhg arzwvg, zvfp adt zvkou wxfhh."
    offset = 7
    key = "devisha"

    first_step = ceasar_decode(string, offset)
    print(first_step)
    print()

    first_step_simpler = "jsjl bvb gixwvpnj xcm elsvebm, qvu’ui v aehrw lvukaeu. pjdw fox. rje, am yry xwewlhxzl loiv fzngye wlz namthiibz vf mygg a vwh cjc ooawiqmj fox avvl. pf bsp asf yry rifa nrxcqfn tkii ggb’rh kzblpnj e gcfjh dry lwzshvo wf te. lj twm joptgmlld lx vnllr wlz ixvrhqzvlpoqiy lsae, wlzv qvu raz uw h crjamw. zoxrya xhiu vdoza? grsy rgi crqktwaiqk opaz tkspoz, soyi twm sodhn pqyaa."
    second_step = vigenere_decode(first_step_simpler, key)
    print(second_step)


if __name__ == "__main__":
    main()