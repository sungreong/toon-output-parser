#!/bin/bash

# ============================================================
# Examples 테스트 스크립트
# ============================================================
# 모든 examples/*.py 파일을 실행하고 결과를 output/*.log에 저장
# ============================================================

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 출력 디렉토리 생성
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# 시작 시간 기록
START_TIME=$(date +%s)

echo "============================================================"
echo "🚀 Examples 테스트 시작"
echo "============================================================"
echo ""

# Python 파일 목록 가져오기 (숫자 순서대로 정렬)
PYTHON_FILES=$(ls -1 *.py 2>/dev/null | sort -V)

if [ -z "$PYTHON_FILES" ]; then
    echo -e "${RED}❌ Python 파일을 찾을 수 없습니다.${NC}"
    exit 1
fi

# 파일 개수 계산
TOTAL_FILES=$(echo "$PYTHON_FILES" | wc -l)
CURRENT=0
SUCCESS=0
FAILED=0

# 각 파일 실행
while IFS= read -r PYTHON_FILE; do
    CURRENT=$((CURRENT + 1))
    
    # 파일명에서 확장자 제거
    BASENAME=$(basename "$PYTHON_FILE" .py)
    LOG_FILE="$OUTPUT_DIR/${BASENAME}.log"
    
    echo -e "${BLUE}[$CURRENT/$TOTAL_FILES]${NC} 실행 중: ${YELLOW}$PYTHON_FILE${NC}"
    
    # 실행 시작 시간
    FILE_START=$(date +%s)
    
    # Python 파일 실행 및 로그 저장
    if python3 "$PYTHON_FILE" > "$LOG_FILE" 2>&1; then
        FILE_END=$(date +%s)
        FILE_DURATION=$((FILE_END - FILE_START))
        
        SUCCESS=$((SUCCESS + 1))
        echo -e "  ${GREEN}✅ 성공${NC} (${FILE_DURATION}초) → ${GREEN}$LOG_FILE${NC}"
    else
        FILE_END=$(date +%s)
        FILE_DURATION=$((FILE_END - FILE_START))
        
        FAILED=$((FAILED + 1))
        echo -e "  ${RED}❌ 실패${NC} (${FILE_DURATION}초) → ${RED}$LOG_FILE${NC}"
        
        # 에러 내용 미리보기 (마지막 3줄)
        echo -e "  ${RED}에러 내용:${NC}"
        tail -n 3 "$LOG_FILE" | sed 's/^/    /' || true
    fi
    
    echo ""
    
done <<< "$PYTHON_FILES"

# 종료 시간 기록
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# 결과 요약
echo "============================================================"
echo "📊 테스트 결과 요약"
echo "============================================================"
echo -e "총 파일 수: ${BLUE}$TOTAL_FILES${NC}"
echo -e "성공: ${GREEN}$SUCCESS${NC}"
echo -e "실패: ${RED}$FAILED${NC}"
echo -e "총 소요 시간: ${BLUE}${TOTAL_DURATION}초${NC}"
echo ""

# 로그 파일 위치
echo "============================================================"
echo "📁 로그 파일 위치: ${BLUE}$OUTPUT_DIR/${NC}"
echo "============================================================"

# 실패한 파일 목록 표시
if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}실패한 파일 목록:${NC}"
    CURRENT=0
    while IFS= read -r PYTHON_FILE; do
        CURRENT=$((CURRENT + 1))
        BASENAME=$(basename "$PYTHON_FILE" .py)
        LOG_FILE="$OUTPUT_DIR/${BASENAME}.log"
        
        # 로그 파일에 에러가 있는지 확인
        if grep -q -i "error\|exception\|traceback\|failed" "$LOG_FILE" 2>/dev/null; then
            echo -e "  ${RED}❌ $PYTHON_FILE${NC} → $LOG_FILE"
        fi
    done <<< "$PYTHON_FILES"
fi

echo ""

# 종료 코드 설정
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ 모든 테스트가 성공적으로 완료되었습니다!${NC}"
    exit 0
else
    echo -e "${RED}❌ 일부 테스트가 실패했습니다.${NC}"
    exit 1
fi
