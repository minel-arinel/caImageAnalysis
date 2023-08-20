

def round_microseconds(str_time):
    '''Rounds the microseconds of Bruker to 6 digits for datetime compatibility'''
    seconds = str_time[str_time.rfind(':')+1:]
    rounded_microsecs = round(float(seconds), 6)

    return str_time[:str_time.rfind(':')+1] + str(rounded_microsecs)