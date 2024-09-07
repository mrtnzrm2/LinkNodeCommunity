s = "A man, a plan, a canal, Panama!"

def checkPalindrome(S : str):
  l = len(S)
  i = 0
  j = l - 1

  while i < l and j > -1:
    if S[i].isalpha() and S[j].isalpha():
      if S[i].lower() != S[j].lower():
        return False
      else:
        i += 1
        j -= 1
      
    elif S[i].isalpha() and not S[j].isalpha():
      j -= 1
    else: i += 1

  return True

print(checkPalindrome(s))
  