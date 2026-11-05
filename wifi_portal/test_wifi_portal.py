import unittest
from wifi_portal import (split_terse, parse_scan, key_mgmt_for, validate_credentials,
                         explain_error)


class T(unittest.TestCase):
    def test_terse_escapes(self):
        self.assertEqual(split_terse(r"My\:Wifi:80:WPA2"), ["My:Wifi", "80", "WPA2"])

    def test_scan_dedupes_and_skips_hidden(self):
        out = "Home:60:WPA2:6\nHome:80:WPA2:11\n:70:WPA2:1\nCafe:40::1"
        r = parse_scan(out)
        self.assertEqual([n["ssid"] for n in r], ["Home", "Cafe"])
        self.assertEqual(r[0]["signal"], 80)
        self.assertTrue(r[1]["open"])

    def test_key_mgmt(self):
        self.assertEqual(key_mgmt_for(""), "open")
        self.assertEqual(key_mgmt_for("--"), "open")
        self.assertEqual(key_mgmt_for("WPA2"), "wpa-psk")
        self.assertEqual(key_mgmt_for("WPA2 WPA3"), "wpa-psk")
        self.assertEqual(key_mgmt_for("WPA3"), "sae")
        self.assertIsNone(key_mgmt_for("WPA2 802.1X"))
        self.assertIsNone(key_mgmt_for("WEP"))

    def test_validate(self):
        self.assertIsNone(validate_credentials("Home", "12345678"))
        self.assertIsNone(validate_credentials("Cafe", ""))
        self.assertIsNotNone(validate_credentials("", "12345678"))
        self.assertIsNotNone(validate_credentials("x" * 33, ""))
        self.assertIsNotNone(validate_credentials("Home", "short"))
        self.assertIsNotNone(validate_credentials("Home", "g" * 64))   # 64 chars must be hex
        self.assertIsNone(validate_credentials("Home", "ab" * 32))

    def test_explain(self):
        self.assertIn("password", explain_error("Error: (7) Secrets were required, but not provided").lower())
        self.assertIn("tidak ditemukan", explain_error("No network with SSID x found").lower())


if __name__ == "__main__":
    unittest.main()
