"""Preprocess cleaned syllabus text files for RAG indexing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


JOINED_WORD_FIXES = {
    "Academicand": "Academic and",
    "academicand": "academic and",
    "AIassistance": "AI assistance",
    "allowyour": "allow your",
    "Anystudentdeemedacademicallydishonestwill": "Any student deemed academically dishonest will",
    "Attendancewill": "Attendance will",
    "Becauseof": "Because of",
    "Becausethis": "Because this",
    "becausetheywill": "because they will",
    "cannothelp": "cannot help",
    "changegrades": "change grades",
    "feedbackwithin": "feedback within",
    "firstthree": "first three",
    "fourthclass": "fourth class",
    "dropcourses": "drop courses",
    "haveattended": "have attended",
    "haveviolated": "have violated",
    "I willtry": "I will try",
    "lectureswhere": "lectures where",
    "meetingbasic": "meeting basic",
    "needsbefore": "needs before",
    "permittedthree": "permitted three",
    "Pleaseread": "Please read",
    "providegrades": "provide grades",
    "showssome": "shows some",
    "viewyour": "view your",
    "worthskipping": "worth skipping",
    "absencebeyond": "absence beyond",
    "absencepenalties": "absence penalties",
    "accommodateyour": "accommodate your",
    "Additionalinformation": "Additional information",
    "anymorethan": "any more than",
    "askingquestions": "asking questions",
    "beforeclass": "before class",
    "beforesubmitting": "before submitting",
    "communicatewith": "communicate with",
    "complywith": "comply with",
    "completeyour": "complete your",
    "conceptsfrom": "concepts from",
    "consultwith": "consult with",
    "courseactivities": "course activities",
    "courseaims": "course aims",
    "coursealso": "course also",
    "coursecombines": "course combines",
    "coursecovering": "course covering",
    "coursecovers": "course covers",
    "coursecoversboth": "course covers both",
    "coursedoes": "course does",
    "coursefocuses": "course focuses",
    "coursegives": "course gives",
    "coursegrade": "course grade",
    "courseintroduces": "course introduces",
    "courseintended": "course intended",
    "coursemanual": "course manual",
    "Coursemanual": "Course manual",
    "coursemanagement": "course management",
    "coursematerial": "course material",
    "coursematerials": "course materials",
    "coursemust": "course must",
    "courseoutline": "course outline",
    "courseperiod": "course period",
    "courseprovides": "course provides",
    "coursereadings": "course readings",
    "courseregarding": "course regarding",
    "courserelated": "course related",
    "courseseeks": "course seeks",
    "coursesome": "course some",
    "coursestudies": "course studies",
    "coursesyllabus": "course syllabus",
    "coursethis": "course this",
    "coursethrough": "course through",
    "coursetotal": "course total",
    "coursetotalgrade": "course total grade",
    "coursewebsite": "course website",
    "coursewhichwill": "course which will",
    "coursewith": "course with",
    "beinghumansamong": "being humans among",
    "otherorganisms": "other organisms",
    "distinctiveabout": "distinctive about",
    "Duringthissurvey": "During this survey",
    "duringthissurvey": "during this survey",
    "thatinclude": "that include",
    "lifethat": "life that",
    "humansneed": "humans need",
    "willdevelop": "will develop",
    "willhave": "will have",
    "willexplore": "will explore",
    "willact": "will act",
    "willalso": "will also",
    "willchoose": "will choose",
    "willcontribute": "will contribute",
    "willcount": "will count",
    "willcover": "will cover",
    "willdiscuss": "will discuss",
    "willinclude": "will include",
    "willlearn": "will learn",
    "willlose": "will lose",
    "willneed": "will need",
    "willnegatively": "will negatively",
    "willrequire": "will require",
    "willresult": "will result",
    "willseriously": "will seriously",
    "willtakeplace": "will take place",
    "willwork": "will work",
    "willwrite": "will write",
    "Thiscourse": "This course",
    "thiscourse": "this course",
    "Thissection": "This section",
    "thissection": "this section",
    "Therewill": "There will",
    "therewill": "there will",
    "Generalcourse": "General course",
    "generalcourse": "general course",
    "Detailedcourse": "Detailed course",
    "detailedcourse": "detailed course",
    "Courseleader": "Course leader",
    "courseleader": "course leader",
    "Courselearning": "Course learning",
    "courselearning": "course learning",
    "coursewill": "course will",
    "courseinstructor": "course instructor",
    "studentswill": "students will",
    "Studentswill": "Students will",
    "studentwill": "student will",
    "Studentwill": "Student will",
    "Studentsshould": "Students should",
    "studentsshould": "students should",
    "studentswith": "students with",
    "willassess": "will assess",
    "willreceive": "will receive",
    "willprovide": "will provide",
    "willintroduce": "will introduce",
    "willfocus": "will focus",
    "willonly": "will only",
    "willbe": "will be",
    "thatmeet": "that meet",
    "thatprovide": "that provide",
    "conditionsthat": "conditions that",
    "behaviorthat": "behavior that",
    "designedsuchthat": "designed such that",
    "readingsfrom": "readings from",
    "assignmentsfrom": "assignments from",
    "helpfrom": "help from",
    "expulsionfrom": "expulsion from",
    "emailwith": "email with",
    "everylecture": "every lecture",
    "higheststandards": "highest standards",
    "fullcitation": "full citation",
    "conjunctionwith": "conjunction with",
    "developedwithin": "developed within",
    "yourinstructor": "your instructor",
    "theirdomain": "their domain",
    "assignmentswill": "assignments will",
    "interfereswith": "interferes with",
    "strugglingwith": "struggling with",
    "intendingthem": "intending them",
    "produceoriginal": "produce original",
    "academichonesty": "academic honesty",
    "distributeyour": "distribute your",
    "payattention": "pay attention",
    "plagiarismwill": "plagiarism will",
    "cheatingwill": "cheating will",
    "yourteaching": "your teaching",
    "lowestpassing": "lowest passing",
    "materialthis": "material this",
    "familiarwith": "familiar with",
    "questionsabout": "questions about",
    "yourclassmates": "your classmates",
    "aboutwhat": "about what",
    "aboutthe": "about the",
    "accordancewith": "accordance with",
    "actwith": "act with",
    "afterclass": "after class",
    "assignmentfrom": "assignment from",
    "assignmentwill": "assignment will",
    "associatedwith": "associated with",
    "basicstandards": "basic standards",
    "checkyour": "check your",
    "checkingyour": "checking your",
    "classesfrom": "classes from",
    "classsessions": "class sessions",
    "Classeswill": "Classes will",
    "classeswill": "classes will",
    "completedbefore": "completed before",
    "contactyour": "contact your",
    "datewill": "date will",
    "dealtwith": "dealt with",
    "deductionfrom": "deduction from",
    "differentfrom": "different from",
    "directlyrelevant": "directly relevant",
    "directyour": "direct your",
    "discussyour": "discuss your",
    "duringoffice": "during office",
    "dismissalfrom": "dismissal from",
    "dropsyourgrade": "drops your grade",
    "engagewith": "engage with",
    "engagementwith": "engagement with",
    "ensureyour": "ensure your",
    "eventthat": "event that",
    "Examrooms": "Exam rooms",
    "everunsure": "ever unsure",
    "examwill": "exam will",
    "examswill": "exams will",
    "examgradewill": "exam grade will",
    "examinehumans": "examine humans",
    "feedbackfrom": "feedback from",
    "finalexamwill": "final exam will",
    "finalgrade": "final grade",
    "finalgradewill": "final grade will",
    "fallingshort": "falling short",
    "fromadditional": "from additional",
    "fromattending": "from attending",
    "fromdesignated": "from designated",
    "fromyour": "from your",
    "givenwith": "given with",
    "goingwell": "going well",
    "gradesreceived": "grades received",
    "gradesthroughout": "grades throughout",
    "gradeswill": "grades will",
    "gradewill": "grade will",
    "guestlectures": "guest lectures",
    "haveachieved": "have achieved",
    "haveperformed": "have performed",
    "havequestions": "have questions",
    "impedeyour": "impede your",
    "improveyour": "improve your",
    "informationabout": "information about",
    "informationsuch": "information such",
    "milestoneswithin": "milestones within",
    "instructionsfrom": "instructions from",
    "instructorwill": "instructor will",
    "interactwith": "interact with",
    "keepchecking": "keep checking",
    "keeptrack": "keep track",
    "knowmore": "know more",
    "latedate": "late date",
    "linewith": "line with",
    "longafter": "long after",
    "makearrangements": "make arrangements",
    "makechanges": "make changes",
    "makeinformed": "make informed",
    "makeindividual": "make individual",
    "makeupexam": "make-up exam",
    "makeupexams": "make-up exams",
    "makesure": "make sure",
    "materialswill": "materials will",
    "meetshigh": "meets high",
    "meetsbasic": "meets basic",
    "meetsmost": "meets most",
    "meetssome": "meets some",
    "missedseveral": "missed several",
    "Moodlewith": "Moodle with",
    "morethan": "more than",
    "Morethan": "More than",
    "NUGrading": "NU Grading",
    "Onlinematerials": "Online materials",
    "otherinformation": "other information",
    "overallgrade": "overall grade",
    "pleasecontact": "please contact",
    "pleasecome": "please come",
    "pleaseemail": "please email",
    "pleasesend": "please send",
    "problemswith": "problems with",
    "readingsprovided": "readings provided",
    "readingswill": "readings will",
    "Refrainfrom": "Refrain from",
    "reportswithyour": "reports with your",
    "reviewthem": "review them",
    "reviewyour": "review your",
    "scoreswill": "scores will",
    "sexualharassment": "sexual harassment",
    "shareyour": "share your",
    "sharingyour": "sharing your",
    "shouldfollow": "should follow",
    "shouldprepare": "should prepare",
    "showyourwork": "show your work",
    "Someclasseswill": "Some classes will",
    "someessential": "some essential",
    "sourcewithout": "source without",
    "studywith": "study with",
    "submityour": "submit your",
    "supplementaryarticle": "supplementary article",
    "takenotes": "take notes",
    "tardinesswill": "tardiness will",
    "testyour": "test your",
    "testingyour": "testing your",
    "thatthings": "that things",
    "thesewill": "these will",
    "Thesewill": "These will",
    "thosemanifest": "those manifest",
    "threeordinary": "three ordinary",
    "Thismeansthat": "This means that",
    "towardsyour": "towards your",
    "violationsfrom": "violations from",
    "violationwill": "violation will",
    "whichwill": "which will",
    "willalways": "will always",
    "whilefalling": "while falling",
    "withoutcustomary": "without customary",
    "withoutgiving": "without giving",
    "withoutproper": "without proper",
    "withoutspecial": "without special",
    "willapprove": "will approve",
    "workwill": "work will",
    "workwith": "work with",
    "workwhen": "work when",
    "yourability": "your ability",
    "yourabsence": "your absence",
    "yourassigned": "your assigned",
    "yourattendance": "your attendance",
    "yourcellphone": "your cellphone",
    "yourcomputer": "your computer",
    "yourcourse": "your course",
    "yourcurrent": "your current",
    "yourfinal": "your final",
    "yourfriends": "your friends",
    "yourgrade": "your grade",
    "yourinitial": "your initial",
    "yourname": "your name",
    "yourneighbors": "your neighbors",
    "yourpeers": "your peers",
    "yourpermanent": "your permanent",
    "yourquestions": "your questions",
    "yourresearch": "your research",
    "youremaildoes": "your email does",
    "youremailwill": "your email will",
    "yourfellow": "your fellow",
    "yournotedoes": "your note does",
    "yourwork": "your work",
}


def split_long_lowercase_token(token: str) -> str:
    """Use wordninja when available, but keep the pipeline runnable without it."""
    if not (len(token) > 14 and token.isalpha() and token.islower()):
        return token

    try:
        import wordninja
    except ImportError:
        return token

    parts = wordninja.split(token)
    if len(parts) >= 2 and all(len(part) >= 3 for part in parts):
        return " ".join(parts)
    return token


PREFIX_SPLITS = {
    "will": {
        "act", "also", "choose", "contribute", "count", "cover", "discuss",
        "focus", "include", "introduce", "learn", "lose", "need", "only",
        "provide", "receive", "require", "result", "work", "write",
    },
    "your": {
        "ability", "absence", "attendance", "cellphone", "classmates",
        "computer", "course", "current", "email", "fellow", "final", "friends",
        "grade", "initial", "instructor", "name", "neighbors", "peers",
        "permanent", "questions", "research", "teaching", "work",
    },
    "course": {
        "activities", "aims", "also", "combines", "covers", "does", "focuses",
        "gives", "grade", "instructor", "introduces", "intended", "leader",
        "learning", "manual", "management", "material", "materials", "must",
        "outline", "period", "provides", "readings", "regarding", "related",
        "seeks", "some", "syllabus", "this", "through", "total", "website",
        "which", "will", "with",
    },
}

SUFFIX_SPLITS = {
    "will": {
        "absence", "absences", "assignment", "assignments", "attendance",
        "cheating", "course", "date", "discussion", "discussions", "exam",
        "examination", "exams", "finalexam", "finalgrade", "grade", "grades",
        "homework", "instructor", "instruction", "misconduct", "penalties",
        "plagiarism", "policy", "readings", "scores", "student", "students",
        "tardiness", "they", "violation", "work",
    },
    "your": {
        "allow", "check", "checking", "complete", "contact", "contest",
        "direct", "discuss", "distribute", "email", "explain", "explaining",
        "from", "impact", "impede", "include", "including", "improve",
        "mustinform", "reflect", "review", "share", "sharing", "show",
        "submit", "test", "towards", "with",
    },
    "with": {
        "accordance", "act", "associated", "class", "communicate", "comply",
        "concepts", "consult", "course", "dealt", "engage", "engagement",
        "given", "ideas", "interact", "line", "problems", "reports", "study",
        "work",
    },
    "from": {
        "absence", "additional", "assignment", "attending", "classes",
        "concepts", "deduction", "designated", "dismissal", "expulsion",
        "feedback", "help", "information", "instructions", "material",
        "readings", "refrain", "students", "violations", "your",
    },
}


def split_known_joined_token(token: str) -> str:
    """Split common PDF-joined tokens while avoiding broad, unsafe edits."""
    match = re.match(r"^([A-Za-z]+)([^A-Za-z].*)?$", token)
    if not match:
        return token

    word = match.group(1)
    suffix = match.group(2) or ""
    lower = word.lower()

    for prefix, endings in PREFIX_SPLITS.items():
        if lower.startswith(prefix) and lower != prefix:
            rest = lower[len(prefix):]
            if rest in endings:
                left = prefix.capitalize() if word[0].isupper() else prefix
                return f"{left} {word[len(prefix):]}{suffix}"

    for joined_suffix, starts in SUFFIX_SPLITS.items():
        if lower.endswith(joined_suffix) and lower != joined_suffix:
            start = lower[: -len(joined_suffix)]
            if start in starts:
                return f"{word[: -len(joined_suffix)]} {joined_suffix}{suffix}"

    return token


def normalize_emails(text: str) -> str:
    """Repair spaces inserted inside common NU email addresses."""
    text = re.sub(r"\b([A-Za-z]+)\.\s+([A-Za-z]+)@", r"\1.\2@", text)
    text = re.sub(r"@nu\.\s+edu\.\s+kz\b", "@nu.edu.kz", text)
    return text


def preprocess(text: str) -> str:
    text = normalize_emails(text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    for source, target in JOINED_WORD_FIXES.items():
        text = text.replace(source, target)

    tokens = [
        split_long_lowercase_token(split_known_joined_token(token))
        for token in text.split(" ")
    ]
    text = " ".join(tokens)

    text = re.sub(r"(?m)^\s*[A-Za-z]\s*$", "", text)
    text = re.sub(r"(?m)^\s*[^\w\s]+\s*$", "", text)
    text = re.sub(r"(?m)^\s*(page\s+\d+|\d+)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def process_file(input_path: Path, output_dir: Path) -> dict[str, object]:
    raw_text = input_path.read_text(encoding="utf-8", errors="ignore")
    processed_text = preprocess(raw_text)
    output_name = input_path.name.replace("_cleaned.txt", "_processed.txt")
    output_path = output_dir / output_name
    output_path.write_text(processed_text, encoding="utf-8")
    return {
        "source": str(input_path),
        "output": str(output_path),
        "before_words": len(raw_text.split()),
        "after_words": len(processed_text.split()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess cleaned syllabus text files.")
    parser.add_argument("--input-dir", default="data/cleaned")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = sorted(input_dir.glob("*.txt"))
    if not text_files:
        raise SystemExit(f"No text files found in {input_dir}")

    print(f"Found {len(text_files)} text files in {input_dir}")
    for input_path in text_files:
        result = process_file(input_path, output_dir)
        print(
            f"{input_path.name}: {result['before_words']} -> "
            f"{result['after_words']} words"
        )


if __name__ == "__main__":
    main()
