//! Minute-resolution UTC scheduling backed by sparse daily bitmaps.
//!
//! Intervals are half-open `[start_minute, end_minute)` Unix-minute ranges.
//! Each occupied day needs 23 `u64`s (184 bytes of bitmap payload), plus the
//! `BTreeMap` overhead. Availability reads allocate nothing. Event mutations
//! restore overlapping reservations when clearing bits; no per-minute counters
//! or per-event copies of the bitmap are needed.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

const MINUTES_PER_DAY: i64 = 1_440;
const WORDS_PER_DAY: usize = 23;
/// Bound individual events and queries to keep work predictable.
pub const MAX_INTERVAL_MINUTES: i64 = 366 * MINUTES_PER_DAY;
pub const MAX_EVENTS: usize = 4_096;
pub const MAX_OCCUPIED_DAYS: usize = 4_096;
pub const MAX_TITLE_BYTES: usize = 1_024;

type DayBitmap = [u64; WORDS_PER_DAY];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Event {
    pub id: u64,
    pub title: String,
    pub start_minute: i64,
    pub end_minute: i64,
}

#[derive(Debug)]
pub struct Schedule {
    events: BTreeMap<u64, Event>,
    days: BTreeMap<i64, DayBitmap>,
    // Zero denotes an exhausted ID space; ordinary IDs begin at one.
    next_id: u64,
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    pub fn new() -> Self {
        Self {
            events: BTreeMap::new(),
            days: BTreeMap::new(),
            next_id: 1,
        }
    }

    pub fn create(
        &mut self,
        title: String,
        start_minute: i64,
        end_minute: i64,
    ) -> Result<Event, String> {
        validate_title(&title)?;
        validate_interval(start_minute, end_minute, false)?;
        if self.events.len() >= MAX_EVENTS {
            return Err(format!("schedule supports at most {MAX_EVENTS} events"));
        }
        self.validate_day_capacity(start_minute, end_minute, None)?;
        if self.next_id == 0 {
            return Err("schedule event ID space is exhausted".into());
        }
        let event = Event {
            id: self.next_id,
            title,
            start_minute,
            end_minute,
        };
        self.next_id = self.next_id.checked_add(1).unwrap_or(0);
        set_occupied(&mut self.days, start_minute, end_minute);
        self.events.insert(event.id, event.clone());
        Ok(event)
    }

    /// Replace an existing event atomically after validating the new interval.
    pub fn update(
        &mut self,
        id: u64,
        title: String,
        start_minute: i64,
        end_minute: i64,
    ) -> Result<Event, String> {
        validate_title(&title)?;
        validate_interval(start_minute, end_minute, false)?;
        let old = self
            .events
            .get(&id)
            .ok_or_else(|| format!("schedule event {id} does not exist"))?;
        let old_start = old.start_minute;
        let old_end = old.end_minute;
        self.validate_day_capacity(start_minute, end_minute, Some(old))?;
        let event = Event {
            id,
            title,
            start_minute,
            end_minute,
        };
        self.events.insert(id, event.clone());
        self.clear_and_restore(old_start, old_end);
        set_occupied(&mut self.days, start_minute, end_minute);
        Ok(event)
    }

    pub fn delete(&mut self, id: u64) -> Option<Event> {
        let event = self.events.remove(&id)?;
        self.clear_and_restore(event.start_minute, event.end_minute);
        Some(event)
    }

    pub fn get(&self, id: u64) -> Option<&Event> {
        self.events.get(&id)
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Return overlapping events ordered by start minute, then ID.
    /// This enumerates stored events; use `is_free` for allocation-free checks.
    pub fn events(&self, start_minute: i64, end_minute: i64) -> Result<Vec<Event>, String> {
        self.events_page(start_minute, end_minute, 0, usize::MAX)
    }

    /// Sort borrowed matching events and clone only the selected page, avoiding
    /// title allocations proportional to the full result set.
    pub fn events_page(
        &self,
        start_minute: i64,
        end_minute: i64,
        offset: usize,
        limit: usize,
    ) -> Result<Vec<Event>, String> {
        validate_interval(start_minute, end_minute, true)?;
        if start_minute == end_minute || limit == 0 {
            return Ok(Vec::new());
        }
        let mut events: Vec<_> = self
            .events
            .values()
            .filter(|event| event.start_minute < end_minute && event.end_minute > start_minute)
            .collect();
        events.sort_unstable_by_key(|event| (event.start_minute, event.id));
        Ok(events
            .into_iter()
            .skip(offset)
            .take(limit)
            .cloned()
            .collect())
    }

    /// Check availability using at most 23 words per touched day, without
    /// allocations or an event scan. An empty interval is free.
    pub fn is_free(&self, start_minute: i64, end_minute: i64) -> Result<bool, String> {
        validate_interval(start_minute, end_minute, true)?;
        for (day, start, end) in DayRanges::new(start_minute, end_minute) {
            if let Some(bitmap) = self.days.get(&day) {
                for (word, mask) in WordMasks::new(start, end) {
                    if bitmap[word] & mask != 0 {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    /// Find the earliest contiguous free run inside the requested interval.
    /// Empty days and words are skipped as a unit; mixed words use bit scans.
    pub fn first_free(
        &self,
        start_minute: i64,
        end_minute: i64,
        duration_minutes: u32,
    ) -> Result<Option<i64>, String> {
        let span = validate_interval(start_minute, end_minute, true)?;
        if duration_minutes == 0 {
            return Err("duration_minutes must be greater than zero".into());
        }
        if i64::from(duration_minutes) > span {
            return Ok(None);
        }

        let mut cursor = start_minute;
        let mut free_start = start_minute;
        let mut free_length = 0_u32;
        for (day, start, end) in DayRanges::new(start_minute, end_minute) {
            let Some(bitmap) = self.days.get(&day) else {
                if free_length == 0 {
                    free_start = cursor;
                }
                free_length += (end - start) as u32;
                if free_length >= duration_minutes {
                    return Ok(Some(free_start));
                }
                cursor += (end - start) as i64;
                continue;
            };

            let mut offset = start;
            while offset < end {
                let word_index = offset / 64;
                let bit_offset = offset % 64;
                let length = (64 - bit_offset).min(end - offset);
                let mut occupied = (bitmap[word_index] >> bit_offset) & low_mask(length);
                let mut remaining = length;
                while remaining != 0 {
                    let free = (occupied.trailing_zeros() as usize).min(remaining);
                    if free != 0 {
                        if free_length == 0 {
                            free_start = cursor;
                        }
                        free_length += free as u32;
                        if free_length >= duration_minutes {
                            return Ok(Some(free_start));
                        }
                        cursor += free as i64;
                        occupied = occupied.checked_shr(free as u32).unwrap_or(0);
                        remaining -= free;
                    }
                    if remaining != 0 {
                        let busy = (occupied.trailing_ones() as usize).min(remaining);
                        cursor += busy as i64;
                        free_length = 0;
                        occupied = occupied.checked_shr(busy as u32).unwrap_or(0);
                        remaining -= busy;
                    }
                }
                offset += length;
            }
        }
        Ok(None)
    }

    /// Return events in ID order for persistence.
    pub fn all_events(&self) -> Vec<Event> {
        self.events.values().cloned().collect()
    }

    /// Restore persisted events and reconstruct occupancy. Duplicate or zero
    /// IDs and malformed intervals are rejected instead of silently overwritten.
    pub fn from_events(events: Vec<Event>) -> Result<Self, String> {
        if events.len() > MAX_EVENTS {
            return Err(format!("schedule supports at most {MAX_EVENTS} events"));
        }
        let mut schedule = Self::new();
        let mut largest_id = 0;
        for event in events {
            validate_title(&event.title)?;
            validate_interval(event.start_minute, event.end_minute, false)?;
            if event.id == 0 {
                return Err("schedule event IDs must be greater than zero".into());
            }
            if schedule.events.contains_key(&event.id) {
                return Err(format!("duplicate schedule event ID {}", event.id));
            }
            schedule.validate_day_capacity(event.start_minute, event.end_minute, None)?;
            largest_id = largest_id.max(event.id);
            set_occupied(&mut schedule.days, event.start_minute, event.end_minute);
            schedule.events.insert(event.id, event);
        }
        schedule.next_id = largest_id.checked_add(1).unwrap_or(0);
        Ok(schedule)
    }

    /// Bitmap payload only; excludes map nodes and event strings.
    pub fn bitmap_bytes(&self) -> usize {
        self.days.len() * std::mem::size_of::<DayBitmap>()
    }

    fn validate_day_capacity(
        &self,
        start_minute: i64,
        end_minute: i64,
        replaced: Option<&Event>,
    ) -> Result<(), String> {
        let added = DayRanges::new(start_minute, end_minute)
            .filter(|(day, _, _)| !self.days.contains_key(day))
            .count();
        if self.days.len() + added <= MAX_OCCUPIED_DAYS {
            return Ok(());
        }
        // Near the limit, account for old days reclaimed by an update. Only
        // this uncommon path scans other events for each old day.
        let mut removed = 0;
        if let Some(old) = replaced {
            let new_first_day = start_minute.div_euclid(MINUTES_PER_DAY);
            let new_last_day = (end_minute - 1).div_euclid(MINUTES_PER_DAY);
            for (day, _, _) in DayRanges::new(old.start_minute, old.end_minute) {
                if (new_first_day..=new_last_day).contains(&day) {
                    continue;
                }
                let shared = self.events.values().any(|event| {
                    event.id != old.id
                        && event.start_minute.div_euclid(MINUTES_PER_DAY) <= day
                        && (event.end_minute - 1).div_euclid(MINUTES_PER_DAY) >= day
                });
                if !shared {
                    removed += 1;
                }
            }
        }
        if self.days.len() + added - removed > MAX_OCCUPIED_DAYS {
            return Err(format!(
                "schedule supports at most {MAX_OCCUPIED_DAYS} occupied days"
            ));
        }
        Ok(())
    }

    fn clear_and_restore(&mut self, start_minute: i64, end_minute: i64) {
        for (day, start, end) in DayRanges::new(start_minute, end_minute) {
            if let Some(bitmap) = self.days.get_mut(&day) {
                for (word, mask) in WordMasks::new(start, end) {
                    bitmap[word] &= !mask;
                }
            }
        }

        // Clearing one event must not free another event's overlapping minutes.
        for event in self.events.values() {
            let start = start_minute.max(event.start_minute);
            let end = end_minute.min(event.end_minute);
            if start < end {
                set_occupied(&mut self.days, start, end);
            }
        }

        // Reclaim bitmap storage when a day no longer contains reservations.
        for (day, _, _) in DayRanges::new(start_minute, end_minute) {
            if self
                .days
                .get(&day)
                .is_some_and(|bitmap| bitmap.iter().all(|word| *word == 0))
            {
                self.days.remove(&day);
            }
        }
    }
}

fn validate_title(title: &str) -> Result<(), String> {
    if title.trim().is_empty() {
        return Err("schedule event title must not be empty".into());
    }
    if title.len() > MAX_TITLE_BYTES {
        return Err(format!(
            "schedule event titles must be at most {MAX_TITLE_BYTES} bytes"
        ));
    }
    Ok(())
}

fn validate_interval(start: i64, end: i64, allow_empty: bool) -> Result<i64, String> {
    let span = end
        .checked_sub(start)
        .ok_or_else(|| "schedule interval length overflows signed minutes".to_owned())?;
    if span < 0 || (!allow_empty && span == 0) {
        return Err(if allow_empty {
            "end_minute must be greater than or equal to start_minute".into()
        } else {
            "end_minute must be greater than start_minute".into()
        });
    }
    if span > MAX_INTERVAL_MINUTES {
        return Err("schedule intervals must be at most 366 days".into());
    }
    Ok(span)
}

fn set_occupied(days: &mut BTreeMap<i64, DayBitmap>, start_minute: i64, end_minute: i64) {
    for (day, start, end) in DayRanges::new(start_minute, end_minute) {
        let bitmap = days.entry(day).or_insert([0; WORDS_PER_DAY]);
        for (word, mask) in WordMasks::new(start, end) {
            bitmap[word] |= mask;
        }
    }
}

fn low_mask(bits: usize) -> u64 {
    if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

/// Split validated absolute ranges without multiplying a day by 1,440, which
/// could overflow for ranges near the endpoints of the signed timestamp space.
struct DayRanges {
    cursor: i64,
    end: i64,
}

impl DayRanges {
    fn new(start: i64, end: i64) -> Self {
        Self { cursor: start, end }
    }
}

impl Iterator for DayRanges {
    type Item = (i64, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let day = self.cursor.div_euclid(MINUTES_PER_DAY);
        let start = self.cursor.rem_euclid(MINUTES_PER_DAY);
        let length = (MINUTES_PER_DAY - start).min(self.end - self.cursor);
        self.cursor += length;
        Some((day, start as usize, (start + length) as usize))
    }
}

struct WordMasks {
    cursor: usize,
    end: usize,
}

impl WordMasks {
    fn new(start: usize, end: usize) -> Self {
        Self { cursor: start, end }
    }
}

impl Iterator for WordMasks {
    type Item = (usize, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.end {
            return None;
        }
        let word = self.cursor / 64;
        let offset = self.cursor % 64;
        let length = (64 - offset).min(self.end - self.cursor);
        self.cursor += length;
        Some((word, low_mask(length) << offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_open_boundaries_and_negative_days() {
        let mut schedule = Schedule::new();
        let event = schedule.create("midnight".into(), -2, 2).unwrap();
        assert_eq!(schedule.bitmap_bytes(), 2 * 184);
        assert!(schedule.is_free(-10, -2).unwrap());
        assert!(!schedule.is_free(-2, -1).unwrap());
        assert!(!schedule.is_free(1, 2).unwrap());
        assert!(schedule.is_free(2, 10).unwrap());
        assert_eq!(schedule.first_free(-5, 5, 3).unwrap(), Some(-5));
        assert_eq!(schedule.first_free(-4, 6, 4).unwrap(), Some(2));
        assert_eq!(schedule.first_free(-4, 5, 4).unwrap(), None);
        assert!(schedule.events(2, 10).unwrap().is_empty());
        assert_eq!(schedule.events(-1, 1).unwrap(), vec![event.clone()]);
        assert_eq!(schedule.delete(event.id), Some(event));
        assert_eq!(schedule.bitmap_bytes(), 0);
    }

    #[test]
    fn deletion_and_update_preserve_overlapping_reservations() {
        let mut schedule = Schedule::new();
        let first = schedule.create("first".into(), 0, 100).unwrap();
        let second = schedule.create("second".into(), 50, 150).unwrap();
        schedule.delete(first.id).unwrap();
        assert!(schedule.is_free(0, 50).unwrap());
        assert!(!schedule.is_free(50, 150).unwrap());
        let third = schedule.create("third".into(), 125, 200).unwrap();
        schedule
            .update(second.id, "moved".into(), -1_441, -1_439)
            .unwrap();
        assert!(schedule.is_free(50, 125).unwrap());
        assert!(!schedule.is_free(125, 200).unwrap());
        assert!(!schedule.is_free(-1_441, -1_439).unwrap());
        assert_eq!(schedule.bitmap_bytes(), 3 * 184);
        schedule.delete(third.id).unwrap();
        assert_eq!(schedule.bitmap_bytes(), 2 * 184);
    }

    #[test]
    fn first_free_crosses_word_and_day_boundaries() {
        let mut schedule = Schedule::new();
        schedule.create("one".into(), 0, 63).unwrap();
        schedule.create("two".into(), 129, 1_438).unwrap();
        schedule.create("three".into(), 1_506, 1_507).unwrap();
        assert_eq!(schedule.first_free(0, 1_507, 66).unwrap(), Some(63));
        assert_eq!(schedule.first_free(0, 1_507, 67).unwrap(), Some(1_438));
        assert_eq!(schedule.first_free(0, 1_507, 69).unwrap(), None);
        // Day 1 is absent; the free run spans an occupied day, an empty day,
        // and the beginning of a later occupied day.
        let mut sparse = Schedule::new();
        sparse.create("left".into(), 0, 1_438).unwrap();
        sparse.create("right".into(), 2_890, 2_900).unwrap();
        assert_eq!(sparse.first_free(0, 2_900, 1_452).unwrap(), Some(1_438));
        assert_eq!(sparse.first_free(0, 2_900, 1_453).unwrap(), None);
    }

    #[test]
    fn bounds_empty_ranges_and_failed_updates_are_safe() {
        let mut schedule = Schedule::new();
        assert!(schedule.create("empty".into(), 0, 0).is_err());
        assert!(schedule.create("reverse".into(), 1, 0).is_err());
        assert!(schedule.is_free(i64::MIN, i64::MAX).is_err());
        assert!(schedule.is_free(0, MAX_INTERVAL_MINUTES + 1).is_err());
        assert!(schedule.is_free(0, 0).unwrap());
        assert!(schedule.events(0, 0).unwrap().is_empty());
        assert_eq!(schedule.first_free(0, 0, 1).unwrap(), None);
        assert!(schedule.first_free(0, 1, 0).is_err());
        for (start, end) in [(i64::MIN, i64::MIN + 10), (i64::MAX - 10, i64::MAX)] {
            let event = schedule.create("edge".into(), start, end).unwrap();
            assert!(!schedule.is_free(start, end).unwrap());
            assert_eq!(schedule.first_free(start, end, 1).unwrap(), None);
            assert!(schedule.update(event.id, "bad".into(), end, start).is_err());
            assert_eq!(schedule.get(event.id), Some(&event));
            schedule.delete(event.id).unwrap();
            assert_eq!(schedule.first_free(start, end, 10).unwrap(), Some(start));
        }
        assert_eq!(schedule.bitmap_bytes(), 0);
    }

    #[test]
    fn persistence_validation_and_id_exhaustion() {
        let mut schedule = Schedule::new();
        schedule.create("a".into(), -1, 5).unwrap();
        schedule.create("b".into(), 3, 10).unwrap();
        let stored = schedule.all_events();
        let mut restored = Schedule::from_events(stored.clone()).unwrap();
        assert_eq!(restored.all_events(), stored);
        assert!(!restored.is_free(-1, 10).unwrap());
        assert_eq!(restored.create("c".into(), 20, 30).unwrap().id, 3);
        assert!(Schedule::from_events(vec![stored[0].clone(), stored[0].clone()]).is_err());
        let mut bad = stored[0].clone();
        bad.id = 0;
        assert!(Schedule::from_events(vec![bad.clone()]).is_err());
        bad.id = 4;
        bad.end_minute = bad.start_minute;
        assert!(Schedule::from_events(vec![bad]).is_err());
        let mut last = stored[0].clone();
        last.id = u64::MAX;
        let mut exhausted = Schedule::from_events(vec![last]).unwrap();
        assert!(exhausted.create("overflow".into(), 0, 1).is_err());
    }

    #[test]
    fn event_pages_have_stable_chronological_order_and_safe_offsets() {
        let mut schedule = Schedule::new();
        let later = schedule.create("later".into(), 30, 40).unwrap();
        let earlier = schedule.create("earlier".into(), 10, 20).unwrap();
        let same_start = schedule.create("same start".into(), 10, 15).unwrap();
        assert_eq!(schedule.events_page(0, 50, 0, 1).unwrap(), vec![earlier]);
        assert_eq!(schedule.events_page(0, 50, 1, 1).unwrap(), vec![same_start]);
        assert_eq!(schedule.events_page(0, 50, 2, 100).unwrap(), vec![later]);
        assert!(schedule
            .events_page(0, 50, usize::MAX, 1)
            .unwrap()
            .is_empty());
        assert!(schedule.events_page(0, 50, 0, 0).unwrap().is_empty());
        assert!(schedule.events_page(50, 0, 0, 0).is_err());
    }

    #[test]
    fn event_and_title_limits_are_enforced_without_mutation() {
        let mut schedule = Schedule::new();
        assert!(schedule.create(" \n".into(), 0, 1).is_err());
        assert!(schedule
            .create("x".repeat(MAX_TITLE_BYTES + 1), 0, 1)
            .is_err());
        for _ in 0..MAX_EVENTS {
            schedule.create("event".into(), 0, 1).unwrap();
        }
        assert_eq!(schedule.len(), MAX_EVENTS);
        assert!(!schedule.is_empty());
        assert!(schedule.create("one too many".into(), 0, 1).is_err());
        assert_eq!(schedule.len(), MAX_EVENTS);
        assert_eq!(schedule.bitmap_bytes(), 184);
        let old = schedule.get(1).unwrap().clone();
        assert!(schedule.update(1, "".into(), 1, 2).is_err());
        assert_eq!(schedule.get(1), Some(&old));
        let mut too_many = schedule.all_events();
        too_many.push(Event {
            id: MAX_EVENTS as u64 + 1,
            ..old
        });
        assert!(Schedule::from_events(too_many).is_err());
    }

    #[test]
    fn bitmap_limit_allows_updates_that_reclaim_old_days() {
        let mut schedule = Schedule::new();
        let mut day = 0;
        while day < MAX_OCCUPIED_DAYS {
            let days = (MAX_OCCUPIED_DAYS - day).min(366);
            schedule
                .create(
                    format!("block {day}"),
                    day as i64 * MINUTES_PER_DAY,
                    (day + days) as i64 * MINUTES_PER_DAY,
                )
                .unwrap();
            day += days;
        }
        assert_eq!(schedule.bitmap_bytes(), MAX_OCCUPIED_DAYS * 184);
        let snapshot = schedule.all_events();
        let extra_start = MAX_OCCUPIED_DAYS as i64 * MINUTES_PER_DAY;
        assert!(schedule
            .create("overflow".into(), extra_start, extra_start + 1)
            .is_err());
        assert_eq!(schedule.all_events(), snapshot);
        let old = schedule.get(1).unwrap().clone();
        let moved_start = extra_start + MINUTES_PER_DAY;
        schedule
            .update(
                1,
                "moved".into(),
                moved_start,
                moved_start + 366 * MINUTES_PER_DAY,
            )
            .unwrap();
        assert_eq!(schedule.bitmap_bytes(), MAX_OCCUPIED_DAYS * 184);
        assert!(schedule.is_free(old.start_minute, old.end_minute).unwrap());
        // The old block is shared with this one-minute event, so moving it
        // again cannot reclaim that day and would exceed the bitmap budget.
        schedule
            .create("shared".into(), moved_start, moved_start + 1)
            .unwrap();
        let before = schedule.get(1).unwrap().clone();
        let beyond = moved_start + 367 * MINUTES_PER_DAY;
        assert!(schedule
            .update(1, "overflow".into(), beyond, beyond + 366 * MINUTES_PER_DAY)
            .is_err());
        assert_eq!(schedule.get(1), Some(&before));
        let mut invalid_snapshot = schedule.all_events();
        invalid_snapshot.push(Event {
            id: u64::MAX,
            title: "extra day".into(),
            start_minute: beyond,
            end_minute: beyond + 1,
        });
        assert!(Schedule::from_events(invalid_snapshot).is_err());
    }

    /// Deterministic randomized mutations compared with a minute-by-minute
    /// independent oracle, including overlapping events and negative minutes.
    #[test]
    fn randomized_mutations_match_reference_oracle() {
        let mut schedule = Schedule::new();
        let mut reference = BTreeMap::<u64, Event>::new();
        let mut state = 0x1a2b_3c4d_5e6f_7788_u64;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for step in 0..1_200 {
            let start = (random() % 6_000) as i64 - 3_000;
            let end = start + (random() % 180 + 1) as i64;
            match random() % 3 {
                0 => {
                    let event = schedule
                        .create(format!("event {step}"), start, end)
                        .unwrap();
                    reference.insert(event.id, event);
                }
                1 if !reference.is_empty() => {
                    let index = random() as usize % reference.len();
                    let id = *reference.keys().nth(index).unwrap();
                    let updated = schedule.update(id, "updated".into(), start, end).unwrap();
                    reference.insert(id, updated);
                }
                2 if !reference.is_empty() => {
                    let index = random() as usize % reference.len();
                    let id = *reference.keys().nth(index).unwrap();
                    assert_eq!(schedule.delete(id), reference.remove(&id));
                }
                _ => {}
            }

            let query_start = (random() % 6_000) as i64 - 3_000;
            let query_end = query_start + (random() % 600) as i64;
            let duration = (random() % 100 + 1) as u32;
            let occupied = |minute| {
                reference
                    .values()
                    .any(|event| event.start_minute <= minute && minute < event.end_minute)
            };
            let expected_free = (query_start..query_end).all(|minute| !occupied(minute));
            assert_eq!(
                schedule.is_free(query_start, query_end).unwrap(),
                expected_free
            );
            let mut expected_first = None;
            let mut run_length = 0;
            for minute in query_start..query_end {
                if occupied(minute) {
                    run_length = 0;
                } else {
                    run_length += 1;
                    if run_length == duration {
                        expected_first = Some(minute + 1 - i64::from(duration));
                        break;
                    }
                }
            }
            assert_eq!(
                schedule
                    .first_free(query_start, query_end, duration)
                    .unwrap(),
                expected_first,
                "first-free mismatch after mutation {step}"
            );
            let mut expected_events: Vec<_> = reference
                .values()
                .filter(|event| {
                    query_start != query_end
                        && event.start_minute < query_end
                        && event.end_minute > query_start
                })
                .cloned()
                .collect();
            expected_events.sort_unstable_by_key(|event| (event.start_minute, event.id));
            assert_eq!(
                schedule.events(query_start, query_end).unwrap(),
                expected_events
            );
        }
    }
}
