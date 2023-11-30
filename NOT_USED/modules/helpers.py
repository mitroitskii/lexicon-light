'''Miscellaneous functions that simplify some things.
'''
from datetime import date

# code from Charlie that makes unique names for runs
def dict_to_id(args):
    if not isinstance(args, dict):
        try:
            args = vars(args)
        except:
            raise Exception

    def _mb_list_to_string(v, k):
        if isinstance(v, list):
            return "@".join(sorted(v))
        elif isinstance(v, bool):
            return k if v else ""
        else:
            return v.replace('/', '') if type(v)==str else v

    # filter out empty string, None, empty list, False.
    def _filter(v):
        return bool(v)

    # def shorten_if_str(v):
    #     if isinstance(v, str):
    #         return v[:3]
    #     else:
    #         return v

    out = {k: _mb_list_to_string(v, k) for k, v in args.items()}
    # out = [f"{k[:2]}:{shorten_if_str(v)}" for k, v in out.items() if _filter(v)]
    out = [f"{v}" for _, v in out.items() if _filter(v)]
    return "-".join(out) + "-" + date.today().strftime('%m%d%H%M%S')
