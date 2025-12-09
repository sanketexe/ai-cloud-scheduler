#!/usr/bin/env python3
"""
FinOps Platform Setup Script

This script helps you set up the FinOps platform for your organization.
It will:
1. Discover AWS accounts
2. Configure cost centers and teams
3. Set up notification channels
4. Create budgets and alerts
5. Initialize cost tracking
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.multi_account_manager import create_multi_account_manager, CrossAccountRole
from backend.core.notification_service import (
    get_notification_service,
    EmailConfig,
    SlackConfig,
    TeamsConfig,
    NotificationMessage,
    NotificationPriority
)
from backend.core.alert_manager import AlertManager, AlertRule
import structlog

logger = structlog.get_logger(__name__)


class FinOpsSetup:
    """Setup wizard for FinOps platform"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config = self.load_config()
        self.account_manager = None
        self.notification_service = get_notification_service()
        self.alert_manager = AlertManager()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(self.config_file):
            print(f"❌ Configuration file not found: {self.config_file}")
            print(f"📝 Please copy config.example.json to {self.config_file} and update it")
            sys.exit(1)
        
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    async def setup_aws_accounts(self):
        """Step 1: Discover and configure AWS accounts"""
        print("\n" + "="*60)
        print("STEP 1: AWS Account Discovery")
        print("="*60)
        
        # Initialize account manager
        aws_config = self.config['aws']['master_account']
        self.account_manager = create_multi_account_manager({
            'access_key_id': aws_config['access_key_id'],
            'secret_access_key': aws_config['secret_access_key'],
            'region': aws_config['region']
        })
        
        print("🔍 Discovering AWS accounts...")
        accounts = await self.account_manager.discover_accounts()
        
        print(f"✅ Discovered {len(accounts)} accounts:")
        for account in accounts:
            print(f"   - {account.account_name} ({account.account_id})")
            if account.team:
                print(f"     Team: {account.team}, Cost Center: {account.cost_center}")
        
        # Tag accounts based on configuration
        print("\n🏷️  Tagging accounts...")
        for team_config in self.config['organization']['teams']:
            team_name = team_config['name']
            cost_center = team_config['cost_center']
            
            # Find accounts that should belong to this team
            # (In production, you'd have a mapping of account IDs to teams)
            for account in accounts:
                if not account.team:  # Only tag untagged accounts
                    # You would implement your logic here to determine which accounts belong to which team
                    pass
        
        print("✅ Account discovery complete")
        return accounts
    
    async def setup_notification_channels(self):
        """Step 2: Configure notification channels"""
        print("\n" + "="*60)
        print("STEP 2: Notification Channels")
        print("="*60)
        
        # Email channels
        if 'email' in self.config['notifications']:
            email_config_data = self.config['notifications']['email']
            
            base_email_config = EmailConfig(
                smtp_host=email_config_data['smtp_host'],
                smtp_port=email_config_data['smtp_port'],
                smtp_username=email_config_data['smtp_username'],
                smtp_password=email_config_data['smtp_password'],
                from_address=email_config_data['from_address'],
                use_tls=email_config_data.get('use_tls', True)
            )
            
            for channel in email_config_data.get('channels', []):
                self.notification_service.register_email_channel(
                    channel['id'],
                    base_email_config
                )
                print(f"✅ Registered email channel: {channel['id']}")
        
        # Slack channels
        if 'slack' in self.config['notifications']:
            for channel in self.config['notifications']['slack'].get('channels', []):
                slack_config = SlackConfig(
                    webhook_url=channel['webhook_url'],
                    channel=channel.get('channel'),
                    username=channel.get('username', 'FinOps Bot'),
                    icon_emoji=channel.get('icon_emoji', ':moneybag:')
                )
                self.notification_service.register_slack_channel(
                    channel['id'],
                    slack_config
                )
                print(f"✅ Registered Slack channel: {channel['id']}")
        
        # Teams channels
        if 'teams' in self.config['notifications']:
            for channel in self.config['notifications']['teams'].get('channels', []):
                teams_config = TeamsConfig(
                    webhook_url=channel['webhook_url']
                )
                self.notification_service.register_teams_channel(
                    channel['id'],
                    teams_config
                )
                print(f"✅ Registered Teams channel: {channel['id']}")
        
        # Test notifications
        print("\n📧 Sending test notifications...")
        test_message = NotificationMessage(
            title='FinOps Platform Setup Complete',
            message='Your FinOps platform has been successfully configured and is ready to use!',
            priority=NotificationPriority.LOW,
            metadata={'setup': True, 'test': True}
        )
        
        # Send to first channel of each type
        test_channels = []
        if 'email' in self.config['notifications']:
            channels = self.config['notifications']['email'].get('channels', [])
            if channels:
                test_channels.append(channels[0]['id'])
        
        if 'slack' in self.config['notifications']:
            channels = self.config['notifications']['slack'].get('channels', [])
            if channels:
                test_channels.append(channels[0]['id'])
        
        if test_channels:
            results = await self.notification_service.send_notification(
                test_channels,
                test_message
            )
            for channel_id, success in results.items():
                if success:
                    print(f"   ✅ Test notification sent to {channel_id}")
                else:
                    print(f"   ❌ Failed to send to {channel_id}")
    
    async def setup_alerts(self):
        """Step 3: Configure alerts"""
        print("\n" + "="*60)
        print("STEP 3: Alert Configuration")
        print("="*60)
        
        # Budget alerts
        for alert_config in self.config['alerts'].get('budget_alerts', []):
            alert_rule = AlertRule(
                rule_id=alert_config['id'],
                name=alert_config['name'],
                condition_type='budget_threshold',
                threshold_value=alert_config['threshold_percentage'],
                resource_filters={'cost_center': [alert_config.get('cost_center', '')]},
                notification_channels=alert_config['notification_channels'],
                cooldown_minutes=alert_config.get('cooldown_minutes', 60),
                enabled=alert_config.get('enabled', True)
            )
            self.alert_manager.add_alert_rule(alert_rule)
            print(f"✅ Created budget alert: {alert_config['name']}")
        
        # Anomaly alerts
        for alert_config in self.config['alerts'].get('anomaly_alerts', []):
            alert_rule = AlertRule(
                rule_id=alert_config['id'],
                name=alert_config['name'],
                condition_type='anomaly',
                threshold_value=alert_config['threshold_percentage'],
                notification_channels=alert_config['notification_channels'],
                cooldown_minutes=alert_config.get('cooldown_minutes', 240),
                enabled=alert_config.get('enabled', True)
            )
            self.alert_manager.add_alert_rule(alert_rule)
            print(f"✅ Created anomaly alert: {alert_config['name']}")
        
        # Waste detection alerts
        for alert_config in self.config['alerts'].get('waste_alerts', []):
            alert_rule = AlertRule(
                rule_id=alert_config['id'],
                name=alert_config['name'],
                condition_type='waste_detection',
                threshold_value=alert_config.get('utilization_threshold_percentage', 5),
                notification_channels=alert_config['notification_channels'],
                cooldown_minutes=alert_config.get('cooldown_minutes', 1440),
                enabled=alert_config.get('enabled', True)
            )
            self.alert_manager.add_alert_rule(alert_rule)
            print(f"✅ Created waste alert: {alert_config['name']}")
    
    async def display_summary(self):
        """Display setup summary"""
        print("\n" + "="*60)
        print("SETUP COMPLETE! 🎉")
        print("="*60)
        
        print("\n📊 Summary:")
        print(f"   - AWS Accounts: {len(self.account_manager.accounts) if self.account_manager else 0}")
        print(f"   - Teams: {len(self.config['organization']['teams'])}")
        print(f"   - Cost Centers: {len(self.config['organization']['cost_centers'])}")
        print(f"   - Notification Channels: {len(self.notification_service.channels)}")
        print(f"   - Alert Rules: {len(self.alert_manager.alert_rules)}")
        
        print("\n🚀 Next Steps:")
        print("   1. Start the application: docker-compose up -d")
        print("   2. Access the frontend: http://localhost:3000")
        print("   3. Access the API docs: http://localhost:8000/docs")
        print("   4. Monitor Celery tasks: docker-compose logs -f worker")
        
        print("\n📚 Documentation:")
        print("   - Setup Guide: SETUP_GUIDE.md")
        print("   - API Documentation: http://localhost:8000/docs")
        print("   - Configuration: config.json")
        
        print("\n⚠️  Important:")
        print("   - Ensure Celery workers are running for scheduled tasks")
        print("   - Cost data syncs every 6 hours by default")
        print("   - Budget monitoring runs every hour")
        print("   - Check logs for any errors: docker-compose logs")
    
    async def run(self):
        """Run the complete setup"""
        print("\n" + "="*60)
        print("FinOps Platform Setup Wizard")
        print("="*60)
        
        try:
            await self.setup_aws_accounts()
            await self.setup_notification_channels()
            await self.setup_alerts()
            await self.display_summary()
            
            print("\n✅ Setup completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            logger.error("Setup failed", error=str(e), exc_info=True)
            return False


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FinOps Platform Setup')
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip sending test notifications'
    )
    
    args = parser.parse_args()
    
    setup = FinOpsSetup(config_file=args.config)
    success = await setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
