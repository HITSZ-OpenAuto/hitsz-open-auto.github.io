import os
import requests
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import certifi
import openai
import yaml
import shutil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from gen_image import generate_image
from dotenv import load_dotenv
from pytz import timezone

# Configuration
load_dotenv()
WEEKDAYS = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
CONFIG = {
    'token': os.getenv('TOKEN'),
    'org_name': os.getenv('ORG_NAME'),
    'news_type': os.getenv('NEWS_TYPE'),
    'openai_key': os.getenv('OPENAI_API_KEY')
}

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

@dataclass
class Commit:
    author: str
    date: datetime.datetime
    message: str
    repo_name: str

class GithubAPI:
    def __init__(self):
        self.session = self._create_session()
        self.headers = {'Authorization': f'token {CONFIG["token"]}'}

    @staticmethod
    def _create_session():
        session = requests.Session()
        retry_strategy = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
        return session

    def fetch_public_repos(self) -> List[Dict]:
        repos = []
        url = f'https://api.github.com/orgs/{CONFIG["org_name"]}/repos'
        
        while url:
            response = self.session.get(url, headers=self.headers)
            if response.status_code == 200:
                repos.extend(repo for repo in response.json() if not repo['private'])
                url = response.links.get('next', {}).get('url')
            else:
                print(f'Failed to fetch repositories: {response.status_code}')
                break
        return repos

    def get_commits(self, repo: str, since: datetime.datetime) -> List[Dict]:
        url = f'https://api.github.com/repos/{CONFIG["org_name"]}/{repo}/commits'
        params = {'since': since.isoformat()}
        response = self.session.get(url, headers=self.headers, params=params)
        return response.json() if response.status_code == 200 else []

class ReportGenerator:
    def __init__(self, github_api: GithubAPI):
        self.github_api = github_api
        self.start_time = self._calculate_start_time()

    def _calculate_start_time(self) -> datetime.datetime:
        delta = datetime.timedelta(weeks=1 if CONFIG['news_type'] == "weekly" else 1)
        return (datetime.datetime.now(datetime.UTC) - delta).replace(
            hour=0, minute=0, second=0, microsecond=0)

    def generate_front_matter(self) -> str:
        data = {
            'title': f"AUTO {'周报' if CONFIG['news_type'] == 'weekly' else '更新速递'}",
            'date': datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
            'authors': [self._get_author_info()],
            'excludeSearch': False,
            'draft': False
        }
        return yaml.dump(data, allow_unicode=True)

    @staticmethod
    def _get_author_info() -> Dict:
        return {
            'name': "ChatGPT" if CONFIG['news_type'] == 'weekly' else "github-actions[bot]",
            'link': "https://github.com/openai" if CONFIG['news_type'] == 'weekly' else "https://github.com/features/actions",
            'image': "https://github.com/openai.png" if CONFIG['news_type'] == 'weekly' else "https://avatars.githubusercontent.com/in/15368"
        }

    def process_commits(self) -> Tuple[List[Commit], Dict[str, str]]:
        repos = self.github_api.fetch_public_repos()
        commits = []
        org_course_name = {}

        for repo in tqdm(repos):
            raw_commits = self.github_api.get_commits(repo['name'], self.start_time)
            if raw_commits and repo['name'] != "hoa-moe":
                commits.extend(self._process_repo_commits(raw_commits, repo['name']))
                if any(c['commit']['author']['name'] != "github-actions" for c in raw_commits):
                    org_course_name[repo['name']] = self._get_repo_name(repo['name'])

        return commits, org_course_name

    def _process_repo_commits(self, raw_commits: List[Dict], repo_name: str) -> List[Commit]:
        return [
            Commit(
                author=commit['commit']['author']['name'],
                date=datetime.datetime.strptime(commit['commit']['author']['date'], 
                    "%Y-%m-%dT%H:%M:%SZ") + datetime.timedelta(hours=8),
                message=commit['commit']['message'],
                repo_name=repo_name
            )
            for commit in raw_commits
        ]

    def _get_repo_name(self, repo_name: str) -> str:
        tag_url = f'https://raw.githubusercontent.com/{CONFIG["org_name"]}/{repo_name}/main/tag.txt'
        response = self.github_api.session.get(tag_url)
        return response.text.split("name:")[1].strip() if response.status_code == 200 else repo_name

    def create_markdown_report(self, commits: List[Commit], org_course_name: Dict[str, str]) -> str:
        """Create a formatted markdown report from commits."""
        commits.sort(key=lambda x: x.date, reverse=True)
        markdown = "## 更新内容\n\n"
        prev_date = None

        for commit in commits:
            if commit.author in ["github-actions", "actions-user"]:
                continue

            if prev_date != commit.date.date():
                markdown += f'### {WEEKDAYS[commit.date.weekday()]} ({commit.date.month}.{commit.date.day})\n\n'
                prev_date = commit.date.date()

            title = org_course_name.get(commit.repo_name, commit.repo_name)
            markdown += (f'- {commit.author} 在 [{title}](https://github.com/{CONFIG["org_name"]}/'
                       f'{commit.repo_name}) 中提交了信息： {commit.message.splitlines()[0]}\n\n')

        return markdown

    def generate_ai_summary(self, report_text: str) -> str:
        """Generate an AI summary using OpenAI's API."""
        print("Generating AI summary...")
        openai.api_key = CONFIG['openai_key']
        openai.base_url = "https://aihubmix.com/v1/"
        
        prompt = f"Generate a summary for the weekly commit report in Chinese:\n\n{report_text}\n\n---\n\nSummary:"
        try:
            completion = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Failed to generate AI summary: {e}")
            return ""

    def save_report(self, report: str) -> None:
        """Save the generated report to the appropriate location."""
        if CONFIG['news_type'] == "daily":
            path = Path('content/news/daily.md')
        else:
            weekly_dir = Path(f'content/news/weekly/weekly-{self.start_time.date()}')
            weekly_dir.mkdir(parents=True, exist_ok=True)
            path = weekly_dir / 'index.md'
            
        path.write_text(report, encoding='utf-8')

    def update_weekly_index(self) -> None:
        """Update the weekly index file with the latest information."""
        if CONFIG['news_type'] == "weekly":
            index_path = Path('content/news/weekly/_index.zh-cn.md')
            current_date = datetime.datetime.now(timezone('Etc/GMT-8')).date()
            
            front_matter = yaml.dump({
                "title": "AUTO 周报",
                "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d"),
                "description": f"AUTO 周报是由 ChatGPT 每周五发布的一份简报，最近更新于 {current_date}。"
            }, allow_unicode=True)
            
            index_path.write_text(f'---\n{front_matter}---\n', encoding='utf-8')

    def generate_report(self, commits: List[Commit], org_course_name: Dict[str, str]) -> None:
        """Generate and save the complete report."""
        markdown_report = self.create_markdown_report(commits, org_course_name)
        final_report = f'---\n{self.generate_front_matter()}---\n\n'

        if CONFIG['news_type'] == "weekly":
            try:
                # Generate and move AI image
                generate_image(CONFIG['openai_key'])

                final_report += '![AI Image of the Week](generated_image_cropped.png)\n\n'
                
                # Generate and add AI summary
                summary = self.generate_ai_summary(markdown_report)
                if summary:
                    final_report += f'## ✨AI 摘要\n\n{summary}\n\n'
            except Exception as e:
                print(f"Failed to generate image or AI summary: {e}")
                final_report += markdown_report

            weekly_dir = Path(f'content/news/weekly/weekly-{self.start_time.date()}')
            for img_name in ['generated_image.png', 'generated_image_cropped.png']:
                shutil.move(img_name, weekly_dir / img_name)
        
        elif CONFIG['news_type'] == "daily":
            final_report += markdown_report

        self.save_report(final_report)
        self.update_weekly_index()

def main():
    print(f'Generating {CONFIG["news_type"]} report...')
    github_api = GithubAPI()
    generator = ReportGenerator(github_api)
    commits, org_course_name = generator.process_commits()

    if not commits:
        print('No commits found in the given period of time')
        return
    
    generator.generate_report(commits, org_course_name)
    print('Report generated successfully')

if __name__ == "__main__":
    main()
