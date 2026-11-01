from ipspot import get_public_ipv4, IPv4API

def get_current_ip_country() -> str:
    """使用 ipspot 一步到位，获取IP并查询其国家代码"""
    # 调用 ip-api.com 作为数据源，并请求地理信息
    result = get_public_ipv4(api=IPv4API.IP_API_COM, geo=True, timeout=10)
    
    if not result.get("status"):
        print(f"Get ip information faild: {result}")
        return ''
    
    data = result["data"]
    country_code = data.get("country_code")
    return country_code