from django.shortcuts import render
from django.db.models import Count, Sum, Q
from django.utils import timezone
from datetime import timedelta

from apps.accounts.decorators import worker_required
from apps.documents.models import Document, OperationLog
from apps.accounts.models import CustomUser


@worker_required
def statistics(request):
    now   = timezone.now()
    month = now - timedelta(days=30)
    week  = now - timedelta(days=7)

    # ── Documents stats ────────────────────────────────────────────────────
    total_docs = Document.objects.count()

    top_viewed = Document.objects.order_by('-total_views')[:10]
    top_downloaded = Document.objects.order_by('-total_downloads')[:10]
    top_edited = Document.objects.order_by('-total_edits')[:10]

    recent_inserts = OperationLog.objects.filter(
        action=OperationLog.INSERT_DOC,
        timestamp__gte=month,
    ).count()

    recent_edits = OperationLog.objects.filter(
        action=OperationLog.EDIT_DOC,
        timestamp__gte=month,
    ).count()

    recent_downloads = OperationLog.objects.filter(
        action=OperationLog.DOWNLOAD_DOC,
        timestamp__gte=month,
    ).count()

    recent_deletions = OperationLog.objects.filter(
        action=OperationLog.DELETE_DOC,
        timestamp__gte=month,
    ).count()

    # Activity over the last 7 days (for sparkline)
    daily_activity = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end   = day_start + timedelta(days=1)
        count = OperationLog.objects.filter(timestamp__gte=day_start, timestamp__lt=day_end).count()
        daily_activity.append({'label': day_start.strftime('%d/%m'), 'count': count})

    ctx = {
        'total_docs': total_docs,
        'top_viewed': top_viewed,
        'top_downloaded': top_downloaded,
        'top_edited': top_edited,
        'recent_inserts': recent_inserts,
        'recent_edits': recent_edits,
        'recent_downloads': recent_downloads,
        'recent_deletions': recent_deletions,
        'daily_activity': daily_activity,
    }

    # ── Admin-only stats ───────────────────────────────────────────────────
    if request.user.is_admin_or_above:
        workers = CustomUser.objects.filter(is_active=True)

        # Who inserted the most
        top_inserters = (
            OperationLog.objects
            .filter(action=OperationLog.INSERT_DOC)
            .values('user__username')
            .annotate(total=Count('id'))
            .order_by('-total')[:10]
        )

        # Who edited the most
        top_editors = (
            OperationLog.objects
            .filter(action=OperationLog.EDIT_DOC)
            .values('user__username')
            .annotate(total=Count('id'))
            .order_by('-total')[:10]
        )

        # Operation log (last 200 entries)
        operation_log = OperationLog.objects.select_related('user').order_by('-timestamp')[:200]

        ctx.update({
            'top_inserters': top_inserters,
            'top_editors': top_editors,
            'operation_log': operation_log,
            'total_workers': workers.count(),
        })

    return render(request, 'stats/statistics.html', ctx)
