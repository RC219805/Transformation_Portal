const supportedRange = ">=20.9.0 <21 || >=22 <26";
const [major = "0", minor = "0"] = process.versions.node.split(".");

const majorNumber = Number.parseInt(major, 10);
const minorNumber = Number.parseInt(minor, 10);

const supported =
  (majorNumber === 20 && minorNumber >= 9) ||
  (majorNumber >= 22 && majorNumber < 26);

if (!supported) {
  console.error(
    `secure-landing-frontdoor requires Node ${supportedRange}. Current runtime: ${process.versions.node}`
  );
  process.exit(1);
}
