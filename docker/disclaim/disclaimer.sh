#!/bin/bash

STATUS="false"
STATUS_CODE=0

FILE=$(realpath "$0")
ROOT=$(dirname "${FILE}")

# read from docs
article_file="${ROOT}/eula.md"
article_content=$(cat "$article_file")

# set how much line we wnat to show
lines_per_page=$(tput lines)

# cal total page
total_lines=$(echo "$article_content" | wc -l)
total_pages=$((total_lines / lines_per_page ))

# init current page
current_page=1

# Color ANIS
RED='\033[1;31m';
BLUE='\033[1;34m';
GREEN='\033[1;32m';
YELLOW='\033[1;33m';
CYAN='\033[1;36m';
NC='\033[0m';

# show page info
display_page() {
    start_line=$(( (current_page - 1) * lines_per_page + 1 ))
    end_line=$(( current_page * lines_per_page ))
    echo -e -n "${YELLOW}"
    echo "$article_content" | sed -n "${start_line},${end_line}p"
    echo -e -n "${NC}"

    # result=$( echo "$article_content" | sed -n "${start_line},${end_line}p")
    # echo -e "\033[43;35m $result \033[0m "
    
}
# show page one
display_page
while true; do

    if [ $current_page -ne $total_pages ]; then

        
        echo -e -n "\033[43;35m ---More-(Press Enter)--- \033[0m "
        read -rsn1 input
        echo -e "\033[1A\033[K"
        
        case "$input" in
            q)
                break
                ;;
            "")
                # show next page
                
                if [ "$current_page" -lt "$total_pages" ]; then
                    current_page=$((current_page + 1))
                    display_page
                fi
                ;;
            *)
                ;;
        esac
    else
        # wait for user response agree or not.
        read response
        if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]
        then
            
            STATUS="true"
            break
        elif [[ $response =~ ^([nN][oO]|[nN])$ ]]
        then
            STATUS="false"
            exit 1
            break

        else
            echo -e -n "${RED}"
            echo -n "Are you agree? [y/N] "
            echo -e -n "${NC}"
        fi
    
    fi
done

echo "$STATUS"

