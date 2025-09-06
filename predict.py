import re
import math
from urllib.parse import urlparse, unquote
from collections import Counter

shorteners = {
    "bit.ly", "goo.gl", "t.co", "ow.ly", "tinyurl.com", "is.gd", "buff.ly", "adf.ly",
    "bitly.com", "cutt.ly", "rebrand.ly", "lnkd.in", "s.id", "youtu.be", "v.gd",
    "shorte.st", "trib.al", "rb.gy"
}

suspicious_tokens = {
    "secure", "account", "login", "verify", "update", "bank", "free", "bonus", "lucky",
    "gift", "promo", "confirm", "ebay", "paypal", "signin", "apple", "microsoft",
    "support", "help", "alert", "unlock", "win", "urgent", "limited", "cancel",
    "invoice", "payment", "wallet"
}

ip_regex = re.compile(
    r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())

def extract_features(u: str):
    try:
        u = unquote(u)
        parsed = urlparse(u if "://" in u else "http://" + u)
        scheme = (parsed.scheme or "").lower()
        netloc = (parsed.netloc or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""

        if "@" in netloc:
            _, _, host = netloc.rpartition("@")
            netloc = host
        host = netloc.split(":")[0]

        port_present = 1 if (":" in netloc and netloc.split(":")[-1].isdigit()) else 0
        host_parts = [p for p in host.split(".") if p]
        subdomain_count = max(len(host_parts) - 2, 0)
        tld = host_parts[-1] if host_parts else ""
        url_no_scheme = u.split("://", 1)[-1]

        digits = sum(ch.isdigit() for ch in u)
        letters = sum(ch.isalpha() for ch in u)
        specials = sum(not ch.isalnum() for ch in u)
        params = len([p for p in query.split("&") if p]) if query else 0
        suspicious_count = sum(1 for tok in suspicious_tokens if tok in u.lower())

        return {
            "url_length": len(u), "host_length": len(host), "path_length": len(path), "query_length": len(query),
            "fragment_length": len(fragment), "count_dots_host": host.count("."), "count_hyphens_host": host.count("-"),
            "count_hyphens_total": u.count("-"), "count_digits": digits, "count_letters": letters, "count_specials": specials,
            "ratio_digits": digits / max(1, len(u)), "ratio_letters": letters / max(1, len(u)), "num_params": params,
            "path_depth": path.count("/"), "has_at_symbol": 1 if "@" in url_no_scheme else 0,
            "has_ip_in_host": 1 if ip_regex.fullmatch(host or "") else 0, "has_port": port_present,
            "is_https_scheme": 1 if scheme == "https" else 0, "https_in_string": 1 if "https" in u.lower() else 0,
            "double_slash_count": u.count("//"), "double_slash_last_pos": u.rfind("//"), "subdomain_count": subdomain_count,
            "tld_length": len(tld), "shortener_domain": 1 if any(s in host for s in shorteners) else 0,
            "suspicious_token_count": suspicious_count, "url_entropy": shannon_entropy(u),
            "host_entropy": shannon_entropy(host), "path_entropy": shannon_entropy(path), "query_entropy": shannon_entropy(query),
            "has_equal_in_query": 1 if "=" in query else 0, "has_exclamation": 1 if "!" in u else 0,
            "has_ip_like_anywhere": 1 if ip_regex.search(u) else 0,
        }
    except Exception as e:
        return {}  # Empty dict on error (handled in API)