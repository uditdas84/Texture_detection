import setuptools

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Texture_detection"
AUTHOR_USER_NAME = "uditdas84"
SRC_REPO = "Texture_detection"
AUTHOR_EMAIL = "uditdas84@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Textture detection",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)