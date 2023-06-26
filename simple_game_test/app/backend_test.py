def send_signal(passed_test):
    if passed_test:
        return {"first":"yay!", "second":"you did it!!"}
    else:
        return {"first":"oh no", "second":"you didn't quite get it..."}