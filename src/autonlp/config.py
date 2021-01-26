HF_AUTONLP_BACKEND_API = "http://autonlp-app-alpha-load-balancer-841182397.us-east-1.elb.amazonaws.com/"


def get_auth_headers(token: str):
    return {"Authorization": f"autonlp {token}"}

