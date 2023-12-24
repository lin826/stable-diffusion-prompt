# Enhancing Stable Video Diffusion Prompts

![overview gif](cheetah.gif) 

> Example article: `In Africa's vast savannah, a swift cheetah races, epitomizing nature's splendor. Its effortless sprint highlights not just its remarkable speed but also the urgent need to protect these majestic creatures and their diminishing habitats.`
> 
> Processed prompts: `best quality,ultra-detailed,masterpiece,hires,8k,In Africa,vast savannah,swift cheetah,races,epitomizing,nature,splendor,effortless,sprint,highlights,remarkable speed,also,urgent need,protect,majestic creatures,diminishing,habitats`

## Prerequirement

Python 3.10.0

### PyTorch

Based on individual OS environments, please download corresponding binary files: https://pytorch.org/get-started/locally/

## Install

```sh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

<!---
### Conda

```sh
conda env create -f requirement.yaml
conda env list
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```
--->

## Results

### 13 Campus News

| Article Title | Webpage Cover Image (baseline) | No Prompt Preprocess | With Prompt Preprocess (ours) |
| ------------- | ------------------------------ | -------------------- | ----------------------------- |
| [BU Students Win $50,000 for Helping L.L.Bean Become More Eco-Friendly](https://www.bu.edu/articles/2023/bu-students-win-bu-sustainability-competition/) | ![](https://www.bu.edu/files/2023/12/Questrom-Sustainability-Competition-winners_feat.jpg) | ![](./outputs/original/gif/BU%20Students%20Win%20$50,000%20for%20Helping%20L.L.Bean%20Become%20More%20Eco-Friendly.gif) | ![](./outputs/gif/BU%20Students%20Win%20$50,000%20for%20Helping%20L.L.Bean%20Become%20More%20Eco-Friendly.gif)
| [Chewy Chocolate Cherry Cookies](https://www.bu.edu/articles/2023/chewy-chocolate-cherry-cookies/) | ![](https://www.bu.edu/files/2023/12/choco-cookies-IMG_2311-feat-crop.jpg) | ![](./outputs/original/gif/Chewy%20Chocolate%20Cherry%20Cookies.gif) | ![](./outputs/gif/Chewy%20Chocolate%20Cherry%20Cookies.gif) |
| [In Town for the Holidays? Here’s How to Make the Most of the Season in Boston](https://www.bu.edu/articles/2023/what-to-do-holidays-in-boston/) | ![](https://www.bu.edu/files/2023/12/iStock-182901347-feat-crop.jpg) | ![](./outputs/original/gif/In%20Town%20for%20the%20Holidays%3F%20Here’s%20How%20to%20Make%20the%20Most%20of%20the%20Season%20in%20Boston.gif) | ![](./outputs/gif/In%20Town%20for%20the%20Holidays%3F%20Here’s%20How%20to%20Make%20the%20Most%20of%20the%20Season%20in%20Boston.gif) |
| [New BU MetroBridge Class Studies the Impact of Gentrification](https://www.bu.edu/articles/2023/new-class-studies-impact-of-gentrification/) | ![](https://www.bu.edu/files/2023/12/23-1644-METRO-003-feat-crop.jpg) | ![](./outputs/original/gif/New%20BU%20MetroBridge%20Class%20Studies%20the%20Impact%20of%20Gentrification.gif) | ![](./outputs/gif/New%20BU%20MetroBridge%20Class%20Studies%20the%20Impact%20of%20Gentrification.gif) |
| [New Gastronomy Course Explores the Political, Historical Ingredients in Latinx Food in the US](https://www.bu.edu/articles/2023/new-gastronomy-course-explores-latinx-food-culture-in-the-us/) | ![](https://www.bu.edu/files/2023/12/Latinx-Food-Course-feat.jpg) | ![](./outputs/original/gif/New%20Gastronomy%20Course%20Explores%20the%20Political,%20Historical%20Ingredients%20in%20Latinx%20Food%20in%20the%20US.gif) | ![](./outputs/gif/New%20Gastronomy%20Course%20Explores%20the%20Political,%20Historical%20Ingredients%20in%20Latinx%20Food%20in%20the%20US.gif) |
| [POV: The Rise of Antisemitism Is Real, and Should Not Be Ignored](https://www.bu.edu/articles/2023/pov-rise-of-antisemitism-should-not-be-ignored/) | ![](https://www.bu.edu/files/2023/12/POV-Rise-of-Antisemitism-leadin.jpg) | ![](./outputs/original/gif/POV:%20The%20Rise%20of%20Antisemitism%20Is%20Real,%20and%20Should%20Not%20Be%20Ignored.gif) | ![](./outputs/gif/POV:%20The%20Rise%20of%20Antisemitism%20Is%20Real,%20and%20Should%20Not%20Be%20Ignored.gif) |
| [Shipwreck, Mutiny, and Murder](https://www.bu.edu/articles/2023/master-thriller-david-grann/) | ![](https://www.bu.edu/files/2023/07/Wager5-feat-crop-copy.jpg) | ![](./outputs/original/gif/Shipwreck,%20Mutiny,%20and%20Murder.gif) | ![](./outputs/gif/Shipwreck,%20Mutiny,%20and%20Murder.gif) |
| [Support These Alumni-Owned Businesses This Holiday Season](https://www.bu.edu/articles/2022/holiday-gift-guide-alumni-owned-businesses/) | ![](https://www.bu.edu/files/2022/12/Alumni-gift-guide-feat.jpg) | ![](./outputs/original/gif/Support%20These%20Alumni-Owned%20Businesses%20This%20Holiday%20Season.gif) | ![](./outputs/gif/Support%20These%20Alumni-Owned%20Businesses%20This%20Holiday%20Season.gif) |
| [Tangled Up in Bob Dylan for 15 Years at BU](https://www.bu.edu/articles/2023/tangled-up-in-bob-dylan-for-15-years-at-bu/) | ![](https://www.bu.edu/files/2023/12/23-1698-BOBDYLAN-002-vert-crop.jpg) | ![](./outputs/original/gif/Tangled%20Up%20in%20Bob%20Dylan%20for%2015%20Years%20at%20BU.gif) | ![](./outputs/gif/Tangled%20Up%20in%20Bob%20Dylan%20for%2015%20Years%20at%20BU.gif) |
| [The Results Are In: These Are the Best Holiday Movies According to Our Instagram Followers](https://www.bu.edu/articles/2023/the-best-holiday-movies-according-to-boston-university-instagram-followers/) | ![](https://www.bu.edu/files/2023/12/Hey-BU-Blog-Headers.jpg) | ![](./outputs/original/gif/The%20Results%20Are%20In:%20These%20Are%20the%20Best%20Holiday%20Movies%20According%20to%20Our%20Instagram%20Followers.gif) | ![](./outputs/gif/The%20Results%20Are%20In:%20These%20Are%20the%20Best%20Holiday%20Movies%20According%20to%20Our%20Instagram%20Followers.gif) |
| [Tracy Schroeder, Transformational Leader of BU IS&T, Departing for West Coast Position](https://www.bu.edu/articles/2023/tracy-schroeder-departing-bu/) | ![](https://www.bu.edu/files/2023/12/Tracy-Schroeder-headshot.jpg) | ![](./outputs/original/gif/Tracy%20Schroeder,%20Transformational%20Leader%20of%20BU%20IS&T,%20Departing%20for%20West%20Coast%20Position.gif) | ![](./outputs/gif/Tracy%20Schroeder,%20Transformational%20Leader%20of%20BU%20IS&T,%20Departing%20for%20West%20Coast%20Position.gif) |
| [US Forest Service’s Dangerous Carbon Disposal Plan Would Foul Our National Forests](https://www.bu.edu/articles/2023/us-forest-services-carbon-disposal-plan-would-foul-our-national-forests/) | ![](https://www.bu.edu/files/2023/12/Stanislaus-National-Forest-leadin.jpg) | ![](./outputs/original/gif/US%20Forest%20Service’s%20Dangerous%20Carbon%20Disposal%20Plan%20Would%20Foul%20Our%20National%20Forests.gif) | ![](./outputs/gif/US%20Forest%20Service’s%20Dangerous%20Carbon%20Disposal%20Plan%20Would%20Foul%20Our%20National%20Forests.gif) |
| [Where to See Boston’s Eight Best Holiday Lights](https://www.bu.edu/articles/2022/see-boston-best-holiday-lights/) | ![](https://www.bu.edu/files/2022/12/feat-holiday-light-thumbnail.jpg) | ![](./outputs/original/gif/Where%20to%20See%20Boston’s%20Eight%20Best%20Holiday%20Lights.gif) | ![](./outputs/gif/Where%20to%20See%20Boston’s%20Eight%20Best%20Holiday%20Lights.gif) |
| []() | ![]() | ![]() | ![]() |

