import re



text = "131116-000-120-CF-123"



pattern = (
    r"^"
    r"(?P<project>\d{4,6})(-)"  # project codes between 4 and 6 numbers 
    r"((?P<zone>\d{3})-)?"      # zone code is optional 
    r"(?P<station>\d{3})-"      #
    r"(?P<unit>\d{2}|CE|CF|CI)-"
    r"(?P<item>\d{3})"
    r"(?P<revision>-[A-Z]{1,2})?"
    r"$"
)

match = re.match(pattern,text)

if match:
    data = match.groupdict()
    print(data)
    print(match.group("project"))

# fucntion to confirm part number matchs Expert standard 
# will return true or false 

# fuc

