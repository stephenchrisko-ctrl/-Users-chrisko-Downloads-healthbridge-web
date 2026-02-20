"""
scheduler.py — Background jobs for Railway deployment
Jobs run inside the FastAPI process (no separate worker needed).
"""

import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("healthbridge.scheduler")

scheduler: AsyncIOScheduler = None
_cached_garmin = None
_cached_peloton = None
_last_sync: datetime = None


def get_cached_data():
    return _cached_garmin, _cached_peloton, _last_sync


def set_cached_data(garmin, peloton):
    global _cached_garmin, _cached_peloton, _last_sync
    _cached_garmin = garmin
    _cached_peloton = peloton
    _last_sync = datetime.now()
    logger.info(f"Cache updated at {_last_sync.isoformat()}")


def init_scheduler(garmin_fn, peloton_fn, email_fn, timezone="America/New_York"):
    global scheduler

    async def _sync():
        logger.info("⏰ Nightly sync starting...")
        g = await garmin_fn()
        p = await peloton_fn()
        set_cached_data(g, p)
        logger.info("✅ Nightly sync complete")

    async def _email():
        logger.info("📧 Sending daily report...")
        await email_fn()

    scheduler = AsyncIOScheduler(timezone=timezone)
    scheduler.add_job(_sync, CronTrigger(hour=2, minute=0), id="nightly_sync",
                      name="Nightly Data Sync", replace_existing=True, misfire_grace_time=3600)
    scheduler.add_job(_email, CronTrigger(hour=10, minute=0), id="daily_email",
                      name="Daily Doctor Email", replace_existing=True, misfire_grace_time=3600)
    scheduler.start()
    logger.info(f"🗓 Scheduler started (tz={timezone}) — sync 2am, email 10am")
    return scheduler
